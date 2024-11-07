# Portrait Image Orientation Classifier Using MobileNet

## Overview
This project implements a portrait orientation classifier using MobileNet architecture via TensorFlow. The system identifies whether a portrait image is rotated (0°, 90°, 180°, or 270°) and automatically corrects it to the standard upright position, utilizing transfer learning for efficient training and deployment.

## Features
- Transfer learning with MobileNet base model
- 4-class orientation classification (0°, 90°, 180°, 270°)
- Automatic orientation correction
- Efficient model with low computational requirements
- Batch processing capabilities

## Requirements
```
tensorflow>=2.x
numpy
opencv-python
matplotlib
scikit-learn
```

## Model Architecture


## Data Preprocessing


## Dataset Structure


## Training Configuration


## Usage Example


## Expected Performance


## Advantages of Using MobileNet
1. **Efficient Architecture**
   - Lightweight model suitable for mobile and embedded devices
   - Fast inference time
   - Small model size

2. **Transfer Learning Benefits**
   - Pre-trained on ImageNet
   - Strong feature extraction capabilities
   - Reduced training time and data requirements
