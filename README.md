# ShiftAddViT: Mixture of Multiplication Primitives Towards Efficient Vision Transformer

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)

**Haoran You***, Huihong Shi*, Yipin Guo* and Yingyan Lin

Accepted by [**NeurIPS 2023**](https://neurips.cc/). More Info:
\[ [**Paper**](https://arxiv.org/abs/2306.06446) | [**Slide**]() | [**Youtube**]() | [**Github**](https://github.com/GATECH-EIC/ShiftAddViT/) \]

---

**Updates**

* We have made the entire code for PVT models publicly available, encompassing training, evaluation, TVM compilation of the entire model, and subsequent throughput measurements and comparisons. For additional information, refer to the `./pvt` directory.
* We have also released the unit test for our MatAdd and MatShift kernels constructed with TVM. This test enables you to replicate the comparison results illustrated in Figures 4 and 5 of our paper. Please refer to the `./Ops_Speedups` folder for more information.
