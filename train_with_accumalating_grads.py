import logging
import os
import warnings
from collections import defaultdict
import random
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torchaudio

import customAudioDataset as data
from customAudioDataset import collate_fn
from losses import disc_loss, total_loss
from model import EncodecModel
from msstftd import MultiScaleSTFTDiscriminator
from scheduler import WarmupCosineLrScheduler
from utils import (count_parameters, save_master_checkpoint, set_seed, start_dist_train)
from balancer import Balancer

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# New method for distributed discriminator training handler
def get_train_discriminator(epoch, config):
    local_train_discriminator = (config.model.train_discriminator 
                                  and epoch >= config.lr_scheduler.warmup_epoch 
                                  and random.random() < float(config.model.train_discriminator))
    
    # Broadcast from cuda:0 to other cuda
    local_train_discriminator_tensor = torch.tensor(int(local_train_discriminator), dtype=torch.int32).cuda()
    if dist.get_rank() == 0:
        local_train_discriminator_tensor.fill_(int(local_train_discriminator))
    dist.broadcast(local_train_discriminator_tensor, src=0)
    
    global_train_discriminator = local_train_discriminator_tensor.item() > 0
    return global_train_discriminator

# Define train one step function with gradient accumulation support
def train_one_step(epoch, optimizer, optimizer_disc, model, disc_model, trainloader, config, scheduler, disc_scheduler, scaler=None, scaler_disc=None, writer=None, balancer=None):
    """
    Train one epoch with gradient accumulation.
    The number of accumulation steps is set via config.optimization.grad_accum_steps.
    The discriminator update condition is evaluated only once per accumulation cycle (using the first mini-batch).
    """
    model.train()
    disc_model.train()
    data_length = len(trainloader)
    
    # Get accumulation steps from config (default 1: no accumulation)
    accum_steps = config.optimization.grad_accum_steps
    
    # Initialize loss accumulators for logging
    accumulated_loss_g = 0.0
    accumulated_losses_g = defaultdict(float)
    accumulated_loss_w = 0.0
    accumulated_loss_disc = 0.0
    
    # This flag will be set at the start of each accumulation cycle.
    update_disc = False

    for idx, input_wav in enumerate(trainloader):
        input_wav = input_wav.contiguous().cuda()  # [B, 1, T]

        # At the beginning of an accumulation cycle, zero gradients and decide on discriminator update.
        if idx % accum_steps == 0:
            optimizer.zero_grad()
            optimizer_disc.zero_grad()
            update_disc = get_train_discriminator(epoch, config)  # Evaluate once per cycle

        # Forward pass with autocast if AMP is enabled
        with autocast(enabled=config.common.amp):
            output, loss_w, _ = model(input_wav)
            logits_real, fmap_real = disc_model(input_wav)
            logits_fake, fmap_fake = disc_model(output)
            losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                input_wav, 
                output, 
                sample_rate=config.model.sample_rate,
            )
            
            # Compute generator loss
            if config.common.amp:
                loss_g = 3 * losses_g['l_g'] + 3 * losses_g['l_feat'] + losses_g['l_t'] / 10 + losses_g['l_f']
                total_loss_gen = (loss_g + loss_w) / accum_steps
            else:
                if balancer is not None:
                    scaled_losses = {k: v / accum_steps for k, v in losses_g.items()}
                    balancer.backward(scaled_losses, output, retain_graph=True)
                    loss_g = sum([losses_g[k] * balancer.weights[k] for k in losses_g.keys()]) / accum_steps
                else:
                    loss_g = (3 * losses_g['l_g'] + 3 * losses_g['l_feat'] + losses_g['l_t'] / 10 + losses_g['l_f']) / accum_steps
                total_loss_gen = loss_g + (loss_w / accum_steps)
        
        # Backward pass for generator
        if config.common.amp:
            scaler.scale(total_loss_gen).backward()
        else:
            if balancer is None:
                ( (3 * losses_g['l_g'] + 3 * losses_g['l_feat'] + losses_g['l_t'] / 10 + losses_g['l_f']) / accum_steps ).backward()
            (loss_w / accum_steps).backward()
        
        # Accumulate logging values (using unscaled loss values)
        if balancer is not None:
            accumulated_loss_g += sum([losses_g[k].item() * balancer.weights[k] for k in losses_g.keys()])
        else:
            accumulated_loss_g += (3 * losses_g['l_g'] + 3 * losses_g['l_feat'] + losses_g['l_t'] / 10 + losses_g['l_f']).item()
        for k, l in losses_g.items():
            accumulated_losses_g[k] += l.item()
        accumulated_loss_w += loss_w.item()
        
        # Use the update_disc flag (determined once per accumulation cycle) for discriminator updates.
        if update_disc:
            with autocast(enabled=config.common.amp):
                logits_real_disc, _ = disc_model(input_wav)
                logits_fake_disc, _ = disc_model(output.detach())  # detach to avoid backprop into generator
                loss_disc = disc_loss(logits_real_disc, logits_fake_disc)
            if config.common.amp:
                scaler_disc.scale(loss_disc / accum_steps).backward()
            else:
                (loss_disc / accum_steps).backward()
            accumulated_loss_disc += loss_disc.item()
        
        # At the end of an accumulation cycle (or at the end of the epoch), update optimizers and schedulers.
        if ((idx + 1) % accum_steps == 0) or (idx == data_length - 1):
            if config.common.amp:
                scaler.step(optimizer)
                scaler.update()
                if update_disc:
                    scaler_disc.step(optimizer_disc)
                    scaler_disc.update()
            else:
                optimizer.step()
                if update_disc:
                    optimizer_disc.step()
            
            scheduler.step()
            disc_scheduler.step()
        
        # Logging (only on rank 0 or non-distributed)
        if (not config.distributed.data_parallel or dist.get_rank() == 0) and (idx % config.common.log_interval == 0 or idx == data_length - 1):
            global_step = (epoch - 1) * data_length + idx
            writer.add_scalar('Train/Loss_G', accumulated_loss_g / (idx + 1), global_step)
            for k, l in accumulated_losses_g.items():
                writer.add_scalar(f'Train/{k}', l / (idx + 1), global_step)
            writer.add_scalar('Train/Loss_W', accumulated_loss_w / (idx + 1), global_step)
            if config.model.train_discriminator and epoch >= config.lr_scheduler.warmup_epoch:
                writer.add_scalar('Train/Loss_Disc', accumulated_loss_disc / (idx + 1), global_step)
                log_msg = (f"Epoch {epoch} {idx+1}/{data_length}\tAvg loss_G: {accumulated_loss_g / (idx + 1):.4f}\t"
                           f"Avg loss_W: {accumulated_loss_w / (idx + 1):.4f}\tAvg loss_Disc: {accumulated_loss_disc / (idx + 1):.4f}\t"
                           f"lr_G: {optimizer.param_groups[0]['lr']:.6e}\tlr_D: {optimizer_disc.param_groups[0]['lr']:.6e}")
            else:
                log_msg = (f"Epoch {epoch} {idx+1}/{data_length}\tAvg loss_G: {accumulated_loss_g / (idx + 1):.4f}\t"
                           f"Avg loss_W: {accumulated_loss_w / (idx + 1):.4f}\t"
                           f"lr_G: {optimizer.param_groups[0]['lr']:.6e}\tlr_D: {optimizer_disc.param_groups[0]['lr']:.6e}")
            logger.info(log_msg)
    

@torch.no_grad()
def test(epoch, model, disc_model, testloader, config, writer):
    model.eval()
    for idx, input_wav in enumerate(testloader):
        input_wav = input_wav.cuda()

        output = model(input_wav)
        logits_real, fmap_real = disc_model(input_wav)
        logits_fake, fmap_fake = disc_model(output)
        loss_disc = disc_loss(logits_real, logits_fake)
        losses_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output)

    if not config.distributed.data_parallel or dist.get_rank() == 0:
        log_msg = (f'| TEST | epoch: {epoch} | loss_g: {sum([l.item() for l in losses_g.values()])} '
                   f'| loss_disc: {loss_disc.item():.4f}')
        for k, l in losses_g.items():
            writer.add_scalar(f'Test/{k}', l.item(), epoch)
        writer.add_scalar('Test/Loss_Disc', loss_disc.item(), epoch)
        logger.info(log_msg)

        # Save a sample reconstruction (not cropped)
        input_wav, _ = testloader.dataset.get()
        input_wav = input_wav.cuda()
        output = model(input_wav.unsqueeze(0)).squeeze(0)
        sp = Path(config.checkpoint.save_folder)
        torchaudio.save(sp / 'GT.wav', input_wav.cpu(), config.model.sample_rate)
        torchaudio.save(sp / 'Reconstruction.wav', output.cpu(), config.model.sample_rate)

def train(local_rank, world_size, config, tmp_file=None):
    logger.handlers.clear()

    # Set up logging handlers
    file_handler = logging.FileHandler(f"{config.checkpoint.save_folder}/train_encodec_bs{config.datasets.batch_size}_lr{config.optimization.lr}.log")
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Set seed if provided
    if config.common.seed is not None:
        set_seed(config.common.seed)

    # Initialize datasets
    trainset = data.CustomAudioDataset(config=config)
    testset = data.CustomAudioDataset(config=config, mode='test')
    
    # Initialize models
    model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, model_norm=config.model.norm, 
        audio_normalize=config.model.audio_normalize,
        segment=eval(config.model.segment), name=config.model.name,
        ratios=config.model.ratios,
    )
    disc_model = MultiScaleSTFTDiscriminator(
        in_channels=config.model.channels,
        out_channels=config.model.channels,
        filters=config.model.filters,
        hop_lengths=config.model.disc_hop_lengths,
        win_lengths=config.model.disc_win_lengths,
        n_ffts=config.model.disc_n_ffts,
    )

    logger.info(model)
    logger.info(disc_model)
    logger.info(config)
    logger.info(f"Encodec Model Parameters: {count_parameters(model)} | Disc Model Parameters: {count_parameters(disc_model)}")
    logger.info(f"model train mode: {model.training} | quantizer train mode: {model.quantizer.training}")

    # Resume training if requested
    resume_epoch = 0
    if config.checkpoint.resume:
        assert config.checkpoint.checkpoint_path != '', "resume path is empty"
        assert config.checkpoint.disc_checkpoint_path != '', "disc resume path is empty"

        model_checkpoint = torch.load(config.checkpoint.checkpoint_path, map_location='cpu')
        disc_model_checkpoint = torch.load(config.checkpoint.disc_checkpoint_path, map_location='cpu')
        model.load_state_dict(model_checkpoint['model_state_dict'])
        disc_model.load_state_dict(disc_model_checkpoint['model_state_dict'])
        resume_epoch = model_checkpoint['epoch']
        if resume_epoch >= config.common.max_epoch:
            raise ValueError(f"resume epoch {resume_epoch} is larger than total epochs {config.common.epochs}")
        logger.info(f"Loaded checkpoint of model and disc_model, resume from {resume_epoch}")

    train_sampler = None
    test_sampler = None
    if config.distributed.data_parallel:
        if config.distributed.init_method == "tmp":
            torch.distributed.init_process_group(
                backend='nccl',
                init_method="file://{}".format(tmp_file),
                rank=local_rank,
                world_size=world_size)
        elif config.distributed.init_method == "tcp":
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", config.common.port)
            distributed_init_method = f"tcp://{master_addr}:{master_port}"
            logger.info(f"distributed_init_method : {distributed_init_method}")
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=distributed_init_method,
                rank=local_rank,
                world_size=world_size)

        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    model.cuda()
    disc_model.cuda()

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.datasets.batch_size,
        sampler=test_sampler,
        shuffle=False, collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory)
    logger.info(f"There are {len(trainloader)} data to train the EnCodec")
    logger.info(f"There are {len(testloader)} data to test the EnCodec")

    # Set optimizers and schedulers
    params = [p for p in model.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': config.optimization.lr}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params': disc_params, 'lr': config.optimization.disc_lr}], betas=(0.5, 0.9))
    scheduler = WarmupCosineLrScheduler(optimizer, max_iter=config.common.max_epoch * len(trainloader), eta_ratio=0.1, warmup_iter=config.lr_scheduler.warmup_epoch * len(trainloader), warmup_ratio=1e-4)
    disc_scheduler = WarmupCosineLrScheduler(optimizer_disc, max_iter=config.common.max_epoch * len(trainloader), eta_ratio=0.1, warmup_iter=config.lr_scheduler.warmup_epoch * len(trainloader), warmup_ratio=1e-4)

    scaler = GradScaler() if config.common.amp else None
    scaler_disc = GradScaler() if config.common.amp else None  

    if config.checkpoint.resume and 'scheduler_state_dict' in model_checkpoint.keys() and 'scheduler_state_dict' in disc_model_checkpoint.keys():
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
        optimizer_disc.load_state_dict(disc_model_checkpoint['optimizer_state_dict'])
        disc_scheduler.load_state_dict(disc_model_checkpoint['scheduler_state_dict'])
        logger.info(f"Loaded optimizer and disc_optimizer state_dict from {resume_epoch}")

    if config.distributed.data_parallel:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        disc_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(disc_model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=config.distributed.find_unused_parameters)
        disc_model = torch.nn.parallel.DistributedDataParallel(
            disc_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=config.distributed.find_unused_parameters)
    if not config.distributed.data_parallel or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=f'{config.checkpoint.save_folder}/runs')
        logger.info(f'Saving tensorboard logs to {Path(writer.log_dir).resolve()}')
    else:
        writer = None

    start_epoch = max(1, resume_epoch + 1)
    balancer_instance = Balancer(dict(config.balancer.weights)) if hasattr(config, 'balancer') else None
    if balancer_instance:
        logger.info(f'Loss balancer with weights {balancer_instance.weights} instantiated')
    test(0, model, disc_model, testloader, config, writer)
    for epoch in range(start_epoch, config.common.max_epoch + 1):
        train_one_step(
            epoch, optimizer, optimizer_disc,
            model, disc_model, trainloader, config,
            scheduler, disc_scheduler, scaler, scaler_disc, writer, balancer_instance)
        if epoch % config.common.test_interval == 0:
            test(epoch, model, disc_model, testloader, config, writer)
        if epoch % config.common.save_interval == 0:
            model_to_save = model.module if config.distributed.data_parallel else model
            disc_model_to_save = disc_model.module if config.distributed.data_parallel else disc_model
            if not config.distributed.data_parallel or dist.get_rank() == 0:
                save_master_checkpoint(epoch, model_to_save, optimizer, scheduler, f'{config.checkpoint.save_location}epoch{epoch}_lr{config.optimization.lr}.pt')
                save_master_checkpoint(epoch, disc_model_to_save, optimizer_disc, disc_scheduler, f'{config.checkpoint.save_location}epoch{epoch}_disc_lr{config.optimization.lr}.pt')

    if config.distributed.data_parallel:
        dist.destroy_process_group()

@hydra.main(config_path='config', config_name='config')
def main(config):
    if config.distributed.torch_distributed_debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    if not os.path.exists(config.checkpoint.save_folder):
        os.makedirs(config.checkpoint.save_folder)
    torch.backends.cudnn.enabled = False
    if config.distributed.data_parallel:
        world_size = config.distributed.world_size
        if config.distributed.init_method == "tmp":
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                start_dist_train(train, world_size, config, tmp_file.name)
        elif config.distributed.init_method == "tcp":
            start_dist_train(train, world_size, config)
    else:
        train(1, 1, config)

if __name__ == '__main__':
    main()
