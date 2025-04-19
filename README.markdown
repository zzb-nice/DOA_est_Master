# A Generic DOA deep learning framework v1

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu118-red.svg)](https://pytorch.org/)    [![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue)](https://doi.org/10.xxxx/xxxxx) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


Most importantly, our code incorporates a universal deep learning framework for DOA estimation, capable of efficiently generating simulation data based on Uniform Linear Array (ULA), Uniform Circular Array (UCA), and other array configurations. It enables the effective extraction of raw data $y_t$, as well as the corresponding sampled covariance matrix (SCM), ideal covariance matrix, signal subspace $U_s$, target DOA values, and the spatial spectrum (SP) associated with the DOA. These data components are utilized to construct comprehensive datasets, which can be leveraged to train deep neural networks for both direct and indirect DOA estimation.

Due to the lower computational efficiency of Python in numerical operations compared to MATLAB, it is essential to adopt batch processing at the code level to enhance computational efficiency and minimize the time overhead associated with iterative loops. The array steering vector $A$, incident signals $s(t)$, and noise $n(t)$ are generated using tensorized methods to maximize efficiency. Furthermore, the processes for generating data such as the sampled covariance matrix (SCM) and spatial spectrum (SP) are fully implemented with batch processing to further improve performance.


<div align='center'>
<img src='https://s2.loli.net/2024/10/11/heZ5HYSMJ7BPufF.png' width='100%' align=center/>
</div>

<!--[](https://s2.loli.net/2024/10/11/heZ5HYSMJ7BPufF.png)-->

Once the dataset has been successfully generated, various deep neural networks can be efficiently designed and their performance tested. In the domain of DOA estimation, there is an expectation to design networks capable of achieving more accurate DOA estimation across diverse configurations.

2.Additionally, to compare the performance of different algorithms, we have implemented subspace-based algorithms such as MUSIC, Root MUSIC, ESPRIT, and Unity ESPRIT, as well as compressed sensing algorithms like $\ell_1$-SVD and SPICE, and deep learning-based algorithms such as SPE-CNN, ASL and Learning-SPICE within our code framework.

3.Furthermore, this repository includes the implementation of the methods presented in the paper [A Deep Learning-Based Supervised Transfer Learning Framework for DOA Estimation with Array Imperfections](文章的URL). The significant contributions of our work can be outlined as follows.

* We propose a novel DOA estimation model based on Vision Transformer, which demonstrates exceptional performance under challenging conditions, such as low SNR and small snapshot scenarios.
* Due to the presence of array imperfections, the data distributions of the source and target domains differ significantly, leading to substantial performance degradation when the model is deployed in practical scenarios. To address this issue, we introduce a transfer learning algorithm to align the features between the source and target domains, enhancing the model's performance in practical scenarios.
* Via extensive simulation, we compare the proposed method with existing approaches across multiple evaluation metrics and demonstrate the superiority of our method in terms of DOA estimation accuracy and robustness.

All codes for simulation and plotting for results presented in the paper are available in this repository. We encourage the use of this code and look forward to further improvements and contributions.
## Citation Information

If the code or methods provided in this project have been helpful for your research or work, please cite the following reference:

> **A Deep Learning-Based Supervised Transfer Learning Framework for DOA Estimation with Array Imperfections**  
> Authors: Bo Zhou, Kaijie Xu, Dan Xu, Mengdao Xing  
> Journal/Conference: To be determined  
> DOI: To be determined

### BibTeX Citation Format
```bibtex
@article{zhou2024doa,
  title     = {A Deep Learning-Based Supervised Transfer Learning Framework for DOA Estimation with Array Imperfections},
  author    = {Bo Zhou, Kaijie Xu, Yinghui Quan and Mengdao Xing},
  journal   = {To be determined},
  year      = {2024},
  volume    = {},
  number    = {},
  pages     = {},
  doi       = {To be determined}
}
```

## Code Usage Notes

**1. Prerequisites** 
Make sure you have installed the required dependencies which is listed in the `environment.yml` file.

Make sure you have installed MATLAB and Python, along with the tools required to call MATLAB functions from Python. For more details, refer to: [https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).
  
**2. Root Directory Set**
You need to confirm that the root directory is correct. The project root directory should be added to Python's search path to correctly import function packages. Additionally, Python's working directory should be set to the current file directory to ensure correct paths for reading and saving models and data.

These configurations are well set by PyCharm. However, if you're using VSCode or other IDEs, there may be a need for additional settings. For vscode, this can be achieved by adding the following content to `./.vscode/settings.json`:
  ```
  {
  "terminal.integrated.env.windows": {
    "PYTHONPATH": "${workspaceFolder}"
  },
  "terminal.integrated.env.linux": {
    "PYTHONPATH": "${workspaceFolder}"
  },
  "terminal.integrated.env.osx": {
    "PYTHONPATH": "${workspaceFolder}"
  }
  "python.terminal.executeInFileDir": true, 
  "code-runner.fileDirectoryAsCwd": true, 
  "terminal.integrated.cwd": "${fileDirname}"
}
  ```

**3. implementation of algorithms**
This repository contains the implementation of various algorithms, all algorithms are implemented through **Python** and **MATLAB**. Because some algorithms require joint execution of MATLAB and Python, you need to carefully adjust the directory and certain code before running it.
all test files are in the test/ directory.
- $\ell_1$-SVD algorithm is implemented in the file **l1_svd.py**, which invokes two files: python_call_l1_SVD_omp_plus.m and python_call_l1_SVD_snap.m. When testing snap variations, you should use python_call_l1_SVD_snap in matlab_l1_svd.predict and comment out python_call_l1_SVD_omp_plus. When testing SNR variations, use python_call_l1_SVD_omp_plus. Otherwise, an error will be triggered.
- Some of our models, such as *Learning-SPICE* and *ASL-2*, require MATLAB for compressed sensing post-processing. After the deep learning-based pre-processing is completed, the corresponding MATLAB code should be executed to obtain the final results.
- When running the training or testing scripts, please ensure that the paths for loading models, loading data, and saving results are correct to ensure smooth code execution.

**4. Available Weights**
All files generated during the execution of our code are uploaded to huggingface: https://huggingface.co/zbb2025/DOA_data_and_results/tree/master.


If you encounter any other problems, you can submit an issue, and I will try to resolve it.
## Description of Files

- **data_creater/**  
The file **data_creater** contains modules for dataset generation and management, primarily consisting of three parts:
  - `signal_datasets/`: Core modules for data generation, storage, and loading operations
  - `create_*_data/`: Scripts for generating angle sets under various configurations
  - `Other Files/`:  Implements additional functionalities

- **article_implement/**  
  Our implementation of state-of-the-art methods, including *SPE-CNN*, *ASL-2*, *SubspaceNet*, and *Learning-SPICE*.

- **models/**  
  Contains the definitions and related code for deep learning models.

- **utils/**  
  Includes various script files for data preprocessing, model training, and evaluation.

- **train/**  
  Contains training scripts for training the model under various SNRs and snapshots conditions.

- **test/**  
  Contains testing scripts, which evaluate our proposed model and compare it with other algorithms, generating loss curves and various evaluation metrics for visualization.

- **matlab_post_process/**  
  Contains MATLAB post-processing scripts that require execution in MATLAB.

- **results/**  
  Stores the model weights and test results.

- **data/**  
  Stores various testing datasets.

- **data_save/**  
  Contains plotting scripts and the final aggregated data.

- **vit_transfer_learning/**  
  Contains the implementation of our proposed transfer learning algorithms.

- **article implement/**  
  Our code implements the methods that presented in the paper.

- **environment.yml**  
  Lists the dependencies and their required versions for the project.

- **README.md**  
  Provides an overview of the project, including usage instructions and guidelines.



