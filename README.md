# ShiftAddViT: Mixture of Multiplication Primitives Towards Efficient Vision Transformer

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)

**Haoran You***, Huihong Shi*, Yipin Guo* and Yingyan Lin

Accepted by [**NeurIPS 2023**](https://neurips.cc/). More Info:
\[ [**Paper**](https://arxiv.org/abs/2306.06446) | [**Slide**](https://neurips.cc/media/neurips-2023/Slides/70751_L4FOulc.pdf) | [**Project**](https://neurips.cc/virtual/2023/poster/70751) | [**Poster**](https://drive.google.com/file/d/1QWsQXQc7hdXKd0WQqu_vTr8wU9833eox/view?usp=sharing) | [**Github**](https://github.com/GATECH-EIC/ShiftAddViT/) \]

---

**Updates**

* We have made the entire code for PVT models publicly available, encompassing training, evaluation, TVM compilation of the entire model, and subsequent throughput measurements and comparisons. For additional information, refer to the `./pvt` directory.
* We have also released the unit test for our MatAdd and MatShift kernels constructed with TVM. This test enables you to replicate the comparison results illustrated in Figures 4 and 5 of our paper. Please refer to the `./Ops_Speedups` folder for more information.

**ToDos**

* Publish the pre-trained checkpoints and provide the corresponding expected TVM output in the form of a `.json` file for replicating our results.
* Upload the presentation to Youtube and share the link.
