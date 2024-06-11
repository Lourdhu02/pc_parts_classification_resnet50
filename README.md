# PC Parts Classification using ResNet50

## Overview

This project aims to classify various PC parts (such as CPUs, GPUs, RAM, motherboards, power supplies, etc.) using a Convolutional Neural Network (CNN) model. The model is built using TensorFlow/Keras and leverages transfer learning with the ResNet50 architecture to achieve high accuracy.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset consists of images of various PC parts. The dataset is split into training, validation, and test sets. Data augmentation techniques are applied to improve the model's generalization.

## Model Architecture

The model is based on the ResNet50 architecture, pre-trained on the ImageNet dataset. The top layers are fine-tuned for the specific task of PC parts classification.

## Training

The model is trained using categorical cross-entropy loss and the Adam optimizer. The training process involves fine-tuning the pre-trained ResNet50 model and further training to improve performance.

## Evaluation

The model is evaluated on a separate test set using metrics such as accuracy, precision, recall, and F1-score. Additionally, confusion matrices and sample predictions are visualized to assess model performance.

## Results

The model achieves high accuracy in classifying various PC parts. Detailed results and visualizations are provided in the [results section](#results).

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Lourdhu02/pc_parts_classification_resnet50.git
    cd pc_parts_classification_resnet50
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run all the cells in Jupyter Notebook

## Visualization

### Training History Plot
Visualize the training and validation accuracy and loss over epochs.

### Confusion Matrix
Generate a confusion matrix to evaluate the model's performance across different classes.

### Sample Predictions
Visualize sample predictions to understand the model's qualitative performance.



---

### Example Code for `requirements.txt`

Ensure you have a `requirements.txt` file for the dependencies:

```
tensorflow
numpy
matplotlib
seaborn
scikit-learn
```
