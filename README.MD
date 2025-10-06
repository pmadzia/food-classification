# Food Classification Project

A deep learning project for food image classification using the Food-101 dataset. This repository explores different CNN architectures, from custom TinyVGG models to transfer learning with EfficientNet variants.

## Project Overview

- Custom dataset creation from Food-101
- Data exploration and visualization
- Model implementation and training
- Data augmentation
- Transfer learning
- Experiment tracking

## Notebooks

**1. Custom dataset**
- Subset selection from Food-101 dataset
- Train/test split

**2. Food classification with TinyVGG**
- Custom CNN architecture implementation
- Baseline training without augmentation
- Enhanced training with data augmentation
- Performance comparison

**3. Food classification with transfer learning and experiment tracking**
- Fine-tuned EfficientNetB0 and EfficientNetV2-S models
- Compared model performance across different configurations

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

- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained models and transforms
- **NumPy& Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **torchinfo**: Model architecture summaries
- **tqdm**: Progress bars
- **TensorBoard**: Experiment tracking and visualization

## Results

- Training/testing loss curves
- Accuracy metrics
- Model comparison

## Citation

```bibtex
@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}
```
