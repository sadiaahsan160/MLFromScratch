# ML_Scratch

![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-required-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-from%20scratch-red)

ML_Scratch is a lightweight **Python machine learning library** built from scratch. It includes implementations of fundamental machine learning algorithms such as **regression models, classification models, neural networks, decision trees, and support vector machines**, along with **multiple optimizers and loss functions**.

The goal of this project is to **understand the internal mechanics of machine learning algorithms** by implementing them manually instead of relying on high-level frameworks.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Requirements](#requirements)
- [Example Usage](#example-usage)
---

## Features

- **Linear Models**
  - Linear Regression
  - Linear Classification
- **Non-Linear Models**
  - Polynomial Regression
  - Non-Linear Classification
- **Decision Trees**
  - Regression Tree
  - Classification Tree
- **Neural Networks**
  - Multi-Layer Perceptron (MLP) for binary and multi-class classification
- **Support Vector Machines (SVM)**
- Custom **optimizers**: Gradient Descent, SGD, Momentum, RMSProp, Adam
- Custom **loss functions**: MSE, MAE, BCE, Accuracy, Error Rate
- Visualization utilities for **loss curves** and **decision boundaries**
- Comparison with scikit-learn models

---

## Repository Structure
```bash
ML_Scratch/
│
├── models/
│   ├── LinearRegression.py
│   ├── LinearClassification.py
│   ├── NonLinearRegression.py
│   ├── NonLinearClassification.py
│   ├── MultiLayerPerceptron.py
│   
│
├── Loss_Functions/
│   └── Losses.py
│
├── Optimizers/
│   └── (optimizer implementations)
│
├── SVM/
│   └── SVM.py
│
├── setup.py
└── README.md
``` 
---

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/SadiaAhsan/ML_Scratch.git
cd ML_Scratch
pip install .
``` 
---

## Requirements

- Python >= 3.6
- numpy
- matplotlib (for plotting)
- scikit-learn (optional, for comparison/testing)

---
## Example Usage
### Linear Classification
```python
from ML_Scratch.models import linear_classification
from Optimizers import Optimizer
optimizer = Optimizer(alpha=0.01, n_iteration=500)

model = linear_classification(optimizer=optimizer)
theta, losses = model.best_fit(X_train, Y_train)
 
predictions = model.predict_class(X_test)
```

### Multi Layer Perceptron
```python
from ML_Scratch.models import MLP
from Optimizers import MLP_Optimizer
optimizer = MLP_Optimizer(
    alpha=0.001,
    n_iteration=500,
    optimizer="adam",
    batch_size=16
)

model = MLP(
    input_size=2,
    hidden_size=10,
    output_size=2,
    optimizer=optimizer,
    lossfunction="BCE"
)

loss = model.best_fit(X_train, Y_train)
predictions = model.forward(X_test)
```

## License

This project is released under the MIT License.

## Educational Purpose

This repository is intended for learning and experimentation. 
The implementations prioritize clarity and understanding over computational efficiency.
