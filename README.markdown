# A Generic DOA deep learning framework v1

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2%2Bcu118-red.svg)](https://pytorch.org/)    [![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue)](https://doi.org/10.xxxx/xxxxx)

Most importantly, our code incorporates a universal deep learning framework for DOA estimation, capable of efficiently generating simulation data based on Uniform Linear Array (ULA), Uniform Circular Array (UCA), and other array configurations. It enables the effective extraction of raw data $y_t$, as well as the corresponding sampled covariance matrix (SCM), ideal covariance matrix, signal subspace $U_s$, target DOA values, and the spatial spectrum (SP) associated with the DOA. These data components are utilized to construct comprehensive datasets, which can be leveraged to train deep neural networks for both direct and indirect DOA estimation.

Due to the lower computational efficiency of Python in numerical operations compared to MATLAB, it is essential to adopt batch processing at the code level to enhance computational efficiency and minimize the time overhead associated with iterative loops. The array steering vector $A$, incident signals $s(t)$, and noise $n(t)$ are generated using tensorized methods to maximize efficiency. Furthermore, the processes for generating data such as the sampled covariance matrix (SCM) and spatial spectrum (SP) are fully implemented with batch processing to further improve performance.


<div align='center'>
<img src='https://s2.loli.net/2024/10/11/heZ5HYSMJ7BPufF.png' width='100%' align=center/>
</div>

<!--[](https://s2.loli.net/2024/10/11/heZ5HYSMJ7BPufF.png)-->

Once the dataset has been successfully generated, various deep neural networks can be efficiently designed and their performance tested. In the domain of DOA estimation, there is an expectation to design networks capable of achieving more accurate DOA estimation across diverse configurations.

2.Additionally, to compare the performance of different algorithms, we have implemented subspace-based algorithms such as MUSIC, Root MUSIC, ESPRIT, and Unity ESPRIT, as well as compressed sensing algorithms like $\ell_1$-SVD, and deep learning-based algorithms such as SPE-CNN within our code framework.

3.Furthermore, this repository includes the implementation of the methods presented in the paper [文章标题](文章的URL).The significant contributions of our work can be outlined as follows.

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
  author    = {Zhou, Bo and Xu, Kaijie and Xu, Dan and Xing, Mengdao},
  journal   = {To be determined},
  year      = {2024},
  volume    = {},
  number    = {},
  pages     = {},
  doi       = {To be determined}
}
```

## Description of Files


- **data_creater/**  
  存放数据集文件，包括原始数据、处理后的数据和实验生成的结果。

- **models/**  
  保存深度学习模型的定义和相关代码。

- **utils/**  
  包含各种脚本文件，用于数据预处理、模型训练和评估。

- **train/**  
  配置文件目录，用于保存训练参数、超参数设置或运行脚本的配置文件。

- **test/**  
  存储训练过程中的日志文件，包括损失值、评估指标和其他调试信息。

- **utils/**  
  工具函数目录，包括通用的辅助函数和模块。

- **results/**  
  保存实验生成的结果，例如输出图片、预测值或对比分析图表。

- **data_save/**  
  存放项目相关的文档，包括项目说明、使用指南等。

- **transfer learning/**  
  包含单元测试代码，确保项目中各模块的正确性。
