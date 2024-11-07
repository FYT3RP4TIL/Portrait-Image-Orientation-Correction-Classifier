# 📷 Portrait Image Orientation Classifier Using MobileNet

## 🌟 Overview
This project implements a portrait orientation classifier using the MobileNet architecture via TensorFlow. The system identifies whether a portrait image is rotated (0°, 90°, 180°, or 270°) and automatically corrects it to the standard upright position, utilizing transfer learning for efficient training and deployment.

## ✨ Features
- ✨ Transfer learning with MobileNet base model
- 🔢 4-class orientation classification (0°, 90°, 180°, 270°)
- 🔄 Automatic orientation correction
- ⚡ Efficient model with low computational requirements
- 🔒 Batch processing capabilities

## 📋 Requirements
```
🐍 tensorflow>=2.x
📚 numpy
🖼️ opencv-python
📊 matplotlib
📏 scikit-learn
```

## 🗂️ Dataset Description
This project utilizes the Kaggle dataset [Deeper1: Rotated Faces Dataset](https://www.kaggle.com/datasets/rodrigov/deeper1/data), which contains a diverse collection of images that have been intentionally rotated in four primary directions: right, left, upside down, and upright. Each image is associated with a label indicating its original orientation.

The dataset characteristics include:
- 🖼️ Image diversity: The dataset encompasses a wide variety of images, including objects, landscapes, people, and other visual elements, ensuring the generalization of the trained model.
- 📐 Accurate rotations: The images have been precisely rotated at 90-degree angles, allowing for clear and objective classification of orientations.
- 📝 Clear labels: Each image is associated with a label indicating its original orientation (right, left, upside down, or upright), facilitating the training process.
- 🔍 Format: The images are in [JPG, PNG] format and have dimensions of [150, 150, 3].

## 🛠️ Data Preprocessing
Before feeding the data into the model, we perform the following preprocessing steps:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Train and validation datagen with rescaling (and optional augmentation for training if needed)
datagen = ImageDataGenerator(rescale=1./255.)

# Separate test datagen, only rescaling
test_datagen_final = ImageDataGenerator(rescale=1./255.)

# Train, validation, and test generators
train_generator = datagen.flow_from_dataframe(...)
valid_generator = datagen.flow_from_dataframe(...)
test_generator_final = test_datagen_final.flow_from_dataframe(...)
```

## 🏛️ MobileNet Model Architecture
The model architecture is based on the MobileNet convolutional neural network, which is known for its efficiency and effectiveness in mobile and embedded devices.

![Dilated-MobileNet-architecture-with-different-dilation-rates-on-its-depthwise](https://github.com/user-attachments/assets/11caf561-49c9-43a3-82db-fb30bc7c7c60)

```python
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt

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

## 🏋️‍♀️ Training Configuration
The MobileNet model is trained using the following configuration:

```python
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

    # Plot training and validation metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, acc, 'b-', label='Training Accuracy')  
    plt.plot(epoch_range, val_acc, 'r-', label='Validation Accuracy')  
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
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

## 📊 Expected Performance
The model is expected to achieve the following performance metrics on the test set:

```
[Loss, Accuracy]
[0.09530360996723175, 0.9701370596885681]

Accuracy: 97.01%
               precision    recall  f1-score   support

 rotated_left       0.95      0.97      0.96      1206
rotated_right       0.97      0.97      0.97      1241
      upright       0.99      0.97      0.98      1223
  upside_down       0.96      0.98      0.97      1219

     accuracy                           0.97      4889
    macro avg       0.97      0.97      0.97      4889
 weighted avg       0.97      0.97      0.97      4889
```

## 🔍 Predictions
The trained model can accurately predict the orientation of portrait images, as shown in the following example:

![1](https://github.com/user-attachments/assets/44acf762-f799-4b35-a93d-f756e37e0bca)

## 🚀 Advantages of Using MobileNet
1. **Efficient Architecture**:
   - 🔍 Lightweight model suitable for mobile and embedded devices
   - ⚡ Fast inference time
   - 💾 Small model size

2. **Transfer Learning Benefits**:
   - 🌐 Pre-trained on ImageNet
   - 🏆 Strong feature extraction capabilities
   - 🕰️ Reduced training time and data requirements

## 📚 Articles and Research Papers
- [Orientation Visualization for Convolutional Neural Networks](https://www.cs.toronto.edu/~guerzhoy/oriviz/crv17.pdf)
- [Automatic Image Orientation Detection Using Deep Learning](https://www.tdcommons.org/cgi/viewcontent.cgi?article=6334&context=dpubs_series)
- [Going Deeper with Convolutions: The Inception Paper Explained](https://medium.com/aiguys/going-deeper-with-convolutions-the-inception-paper-explained-841a0c661fd3)
- [Unleashing the Power of MobileNet: A Comparison with Simple Convolutional Neural Networks](https://medium.com/@zaidbinmuzammil123/unleashing-the-power-of-mobilenet-a-comparison-with-simple-convolutional-neural-networks-71d49f8c86ef)
