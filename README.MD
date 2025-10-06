# Food Classification Project

A deep learning project for food image classification using the Food-101 dataset. This repository explores various CNN architectures, from custom TinyVGG models to transfer learning with EfficientNet variants.

## Project Overview

- Custom dataset creation from Food-101
- Data exploration and visualization
- Model implementation and training
- Data augmentation techniques
- Transfer learning
- Experiment tracking

## Notebooks

1. Custom dataset
- Subset selection from Food-101 dataset
- Train/test split

2. Food classification with TinyVGG
- Custom CNN architecture implementation
- Baseline training without augmentation
- Enhanced training with data augmentation
- Performance comparison

3. Food classification with transfer learning and experiment tracking
- Training modify EfficientNetB0 and EfficientNetV2-S
- Models comparison

## Quick start

Install all required packages:
```sh
pip install -r requirements.txt
```

## Training a model

Run the training script in main directory for basic TinyVGG model:
```sh
python train.py
```
Adjust hyperparameters in train.py for diffrent results or explor the notebooks step-by-step for diffrent implementation.

## Technologies

- PyTorch 2.8.0: Deep learning framework
- torchvision 0.23.0: Pre-trained models and transforms
- NumPy 2.3.3 & Pandas 2.3.3: Data manipulation
- Matplotlib 3.10.6: Visualization
- torchinfo 1.8.0: Model architecture summaries
- tqdm 4.67.1: Progress bars
- TensorBoard 2.20.0: Experiment tracking and visualization

## Results

- Training/testing loss curves
- Accuracy metrics
- Model comparison

## Citation

@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}
