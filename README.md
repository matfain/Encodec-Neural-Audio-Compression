# Encodec Neural Audio Compression

This project implements the Encodec model, a neural audio codec designed for high-quality and efficient audio compression as presented in the [original paper](https://arxiv.org/pdf/2210.13438). Encodec is developed by FAIR (Facebook AI Research) and leverages deep learning to compress and reconstruct audio signals with minimal loss in quality.  

For a detailed review of our re-implementation process and more information about the model architecture you can check our [full report](INSERT LINK).

## Model Demonstrations & Evaluation Notebook
In the evaluation directory, you can find a Jupyter notebook that demonstrates our trained Encodec model's performance on various audio samples. The notebook includes code snippets to load the model, compress audio files, and reconstruct the compressed audio signals. You can use this notebook to evaluate the model's performance on our or yours audio samples.
![img.png](img.png)
![img_1.png](img_1.png)

## Setup Instructions

Follow these steps to set up the project locally:

1. **Clone the repository**
   ```sh
   git clone https://github.com/matfain/Encodec-Neural-Audio-Compression.git
   cd Encodec-Neural-Audio-Compression
   ```

2. **Create a Conda environment**
   * make sure you have Anaconda or Miniconda installed, if not:
     https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
   ```sh
   conda create --name encodec_env python=3.11
   conda activate encodec_env
   ```

4. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
   *   note that this might take a couple of mintues

5. **Download evaluation models**
    - Download the `evaluations.zip` file from [Google Drive](https://drive.google.com/file/d/1AhHsBFfmj2efsI7fKcdwH7D9ewFyr0gE/view?usp=sharing).
    - Extract the contents and place the `evaluations` folder in the project root.

6. **Run the evaluation notebook**
   * enable the encodec_env in the new folder
   ```
   cd evaluations
   ```

   *   Open `evaluations/model_demonstrations.ipynb` in Jupyter Notebook or Jupyter Lab and run the cells to test the model.
   * make sure to choose the 'encodec_env' as the kernel / environment 
   
Enjoy experimenting with neural audio compression! 🚀

## Acknowledgement  
The great majority of the training code is based on the [attached repository](https://github.com/ZhikangNiu/encodec-pytorch). 
