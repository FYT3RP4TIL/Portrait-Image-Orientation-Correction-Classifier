# Portrait Image Orientation Classifier 

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

## Dataset Description

Kaggle - https://www.kaggle.com/datasets/rodrigov/deeper1/data

Description:

This dataset contains a diverse collection of images that have been intentionally rotated in four primary directions: right, left, upside down, and upright. Each image is associated with a label indicating its original orientation. The primary goal of this dataset is to serve as a foundation for the development and training of deep learning models, specifically convolutional neural networks (CNNs), with the purpose of classifying the orientation of each image and consequently correcting its rotation to the upright position.

Dataset Characteristics:

Image diversity: The dataset encompasses a wide variety of images, including objects, landscapes, people, and other visual elements, ensuring the generalization of the trained model.
Accurate rotations: The images have been precisely rotated at 90-degree angles, allowing for clear and objective classification of orientations.
Clear labels: Each image is associated with a label indicating its original orientation (right, left, upside down, or upright), facilitating the training process.
Format: Images are in [specify format, e.g., JPG, PNG] format and have the following dimensions [specify dimensions].


## Data Preprocessing
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Train and validation datagen with rescaling (and optional augmentation for training if needed)
datagen = ImageDataGenerator(rescale=1./255.)

# Separate test datagen, only rescaling
test_datagen_final = ImageDataGenerator(rescale=1./255.)

# Train generator
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="/root/.cache/kagglehub/datasets/rodrigov/deeper1/versions/1/train_rotfaces/train",
    x_col="fn",
    y_col="label",
    batch_size=32,
    seed=42,
    classes=['rotated_left', 'rotated_right', 'upright', 'upside_down'],
    shuffle=True,
    class_mode="categorical",
    target_size=(150, 150)
)

# Validation generator
valid_generator = datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory="/root/.cache/kagglehub/datasets/rodrigov/deeper1/versions/1/train_rotfaces/train",
    x_col="fn",
    y_col="label",
    batch_size=32,
    seed=42,
    shuffle=True,
    classes=['rotated_left', 'rotated_right', 'upright', 'upside_down'],
    class_mode="categorical",
    target_size=(150, 150)
)

# Test generator (using a portion of train data as test set)
# Test generator (with labels included)
test_generator_final = test_datagen_final.flow_from_dataframe(
    dataframe=test_df,
    directory="/root/.cache/kagglehub/datasets/rodrigov/deeper1/versions/1/train_rotfaces/train",  # Same directory as train
    x_col="fn",   # Image filenames
    y_col="label",  # Labels
    target_size=(150, 150),
    batch_size=32,
    shuffle=False,  # No need to shuffle for test data
    class_mode="categorical"  # Ensure this is set to categorical to return labels
)
```


# Mobile-Net Model Architecture
```python
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt

# Create MobileNet model
def create_mobilenet_model():
    input_layer = Input(shape=(150, 150, 3))
    mobilenet = MobileNet(include_top=False, weights='imagenet', input_tensor=input_layer)

    x = Flatten()(mobilenet.output)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(4, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    for layer in mobilenet.layers:
        layer.trainable = False

    adam = Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print("Model Compiled Successfully.")
    print(model.summary())

    return model
```

## Training Configuration

```python
# Train MobileNet model
def train_model(model, train_generator, valid_generator, filepath, epochs=10, batch_size=300):
    print("Starting Model Training.")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    n_train = 39117
    batch_size = 300
    n_valid = 9779

    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=n_train // batch_size,
            epochs=epochs,
            validation_data=valid_generator,
            validation_steps=n_valid // batch_size,
            callbacks=callbacks_list
        )
        print("Model Training Completed Successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    # Plot
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = range(1, len(acc) + 1)

   
    plt.plot(epoch_range, acc, 'b-', label='Training Accuracy')  
    plt.plot(epoch_range, val_acc, 'r-', label='Validation Accuracy')  
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.figure()

    plt.plot(epoch_range, loss, 'b-', label='Training Loss')  
    plt.plot(epoch_range, val_loss, 'r-', label='Validation Loss')  
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    return model
```
![download](https://github.com/user-attachments/assets/78412413-0c11-4251-833d-3c823e249c83)
![download](https://github.com/user-attachments/assets/238361db-d190-46a8-bd14-bcbda0612fef)


## Expected Performance

[Loss, Accuracy]
[0.09530360996723175, 0.9701370596885681]

## Predictions

![1](https://github.com/user-attachments/assets/44acf762-f799-4b35-a93d-f756e37e0bca)

## Advantages of Using MobileNet
1. **Efficient Architecture**
   - Lightweight model suitable for mobile and embedded devices
   - Fast inference time
   - Small model size

2. **Transfer Learning Benefits**
   - Pre-trained on ImageNet
   - Strong feature extraction capabilities
   - Reduced training time and data requirements
