<p align="center">
    <img src="docs/source/figures/logo.png" width="70%" align="center">
</p>

# What is MFBML?
---
[**Documentation**](https://bessagroup.github.io/mfbml/)
| [**Installation**](https://bessagroup.github.io/mfbml/get_started.html)
| [**GitHub**](https://github.com/bessagroup/mfbml)
| [**Tutorials**](https://github.com/bessagroup/mfbml/tree/main/tutorials)

---

### **Summary**

**mfbml** provides provide a general Multi-Fidelity Bayesian Machine Learning framework. The developed MF-BML framework can be used to handle both data scarce and data rich data set scenario depending on the employed algorithm within the framework. The developed MF-BML framework doesn't restrict any algorithm, two configurations are recommended in this repo for handling data scarce and large data set problems respectively.

---

### **State of need**

`mfbml` is a package that supports general multi-fidelity Bayesian machine learning. Two practical multi-fidelity Bayesian machine learning algorithms from the <a href="https://arxiv.org/abs/2407.15110" target="_blank">paper</a>: 1) Kernel Ridge Regression + Linear Transfer-learning + Gaussian Process Regression (KRR-LR-GPR), implemented based on [Numpy](https://numpy.org/); 2) Deep Neural Network + Linear Transfer-learning + Bayesian Neural Network (DNN-LR-BNN), implemented based on [Pytorch](https://pytorch.org/). 

In the particular case of a research environment, `mfbml is designed to easily accommodate further developments, either by improving the already implemented methods or by including new numerical models and techniques.

---

### **Authorship and Citation**

**Author**:

- Jiaxiang Yi ([J.Yi@tudelft.nl](mailto:J.Yi@tudelft.nl))

**Author affiliation**:

- Delft University of Technology

**arXiv** ([paper](https://arxiv.org/abs/2407.15110)):
```
@misc{yi2024practicalmultifidelitymachinelearning,
      title={Practical multi-fidelity machine learning: fusion of deterministic and Bayesian models}, 
      author={Jiaxiang Yi and Ji Cheng and Miguel A. Bessa},
      year={2024},
      eprint={2407.15110},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.15110}, 
}
```

### Get started

**Installation**

(1). git clone the repo to your local machine

```
https://github.com/JiaxiangYi96/mfbml.git
cd mfbml
```
(2) create a new conda environment with python version 3.10 
```
conda create -n mfbml_env python=3.10
conda activate mfbml_env
```
(3). install dependencies first (a git repo [mfpml](https://github.com/JiaxiangYi96/mfpml) with branch `yaga` and [pytorch](https://pytorch.org/get-started/locally/) with `cpu` installation)

```
pip install -r requirements.txt 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

(4). go to the local folder where you cloned the repo, and pip install it with editable mode

```
pip install --verbose --no-build-isolation --editable .
```

---

**Illustrative examples**

1. Kernel Ridge Regression + Linear Transfer-learning + Gaussian Process Regression [Notebook](https://github.com/bessagroup/mfbml/blob/main/tutorials/krr_lr_gpr_illustration.ipynb)


2. Deep Neural Network + Linear Transfer-learning + Bayesian Neural Network [Notebook](https://github.com/bessagroup/mfbml/blob/main/tutorials/mf_dnn_bnn_showcase.ipynb)

3. More illustrative examples shown in the paper can be found in the [studies](https://github.com/bessagroup/mfbml/tree/main/studies) folder.



### **Community Support**

If you have any question, please raise an issue on GitHub or contact the developer

### **License**

BSD 3-Clause License, Jiaxiang Yi

All rights reserved.

mfbml is a free and open-source repo published under BSD 3-Clause License.
