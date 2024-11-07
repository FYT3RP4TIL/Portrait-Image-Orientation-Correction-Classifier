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

# Initialize ImageDataGenerator for training and validation (rescaling only)
datagen = ImageDataGenerator(rescale=1./255.)

# Separate ImageDataGenerator for test data, with only rescaling
test_datagen_final = ImageDataGenerator(rescale=1./255.)

# Training data generator with data augmentation and shuffling
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,  # DataFrame with filenames and labels for training
    directory="/root/.cache/kagglehub/datasets/rodrigov/deeper1/versions/1/train_rotfaces/train",
    x_col="fn",  # Column with image filenames
    y_col="label",  # Column with image labels
    batch_size=32,  # Number of samples per batch
    seed=42,  # Seed for reproducibility
    classes=['rotated_left', 'rotated_right', 'upright', 'upside_down'],
    shuffle=True,  # Shuffle data for better training
    class_mode="categorical",  # Multi-class classification
    target_size=(150, 150)  # Resize all images to 150x150
)

# Validation data generator with similar setup as training (no shuffling)
valid_generator = datagen.flow_from_dataframe(
    dataframe=valid_df,  # DataFrame for validation data
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

# Test data generator without shuffling (evaluating model performance)
test_generator_final = test_datagen_final.flow_from_dataframe(
    dataframe=test_df,  # DataFrame for test data
    directory="/root/.cache/kagglehub/datasets/rodrigov/deeper1/versions/1/train_rotfaces/train",
    x_col="fn",
    y_col="label",
    target_size=(150, 150),
    batch_size=32,
    shuffle=False,  # Keep order for consistent evaluation
    class_mode="categorical"
)
```

## 🏛️ Mobile-Net Model Architecture
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

# Function to create and compile the MobileNet model
def create_mobilenet_model():
    # Define the input layer with a shape suitable for MobileNet (150x150 RGB image)
    input_layer = Input(shape=(150, 150, 3))
    
    # Load the MobileNet model, excluding the fully connected top layers (for feature extraction)
    # Pre-trained weights on ImageNet are used
    mobilenet = MobileNet(include_top=False, weights='imagenet', input_tensor=input_layer)

    # Flatten the output from the last layer of MobileNet for input into fully connected layers
    x = Flatten()(mobilenet.output)
    # Add a dense layer with 256 units and ReLU activation for intermediate processing
    x = Dense(256, activation='relu')(x)
    # Final output layer with 4 units and softmax activation for multiclass classification (4 classes)
    output_layer = Dense(4, activation='softmax')(x)

    # Define the complete model linking input and output layers
    model = Model(inputs=input_layer, outputs=output_layer)

    # Freeze all layers in the MobileNet base model to retain pre-trained features
    for layer in mobilenet.layers:
        layer.trainable = False

    # Set up the Adam optimizer with a low learning rate for fine-tuning
    adam = Adam(learning_rate=1e-4)
    # Compile the model with categorical cross-entropy loss and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # Confirmation of successful model compilation and summary of model architecture
    print("Model Compiled Successfully.")
    print(model.summary())

    return model
```

## 🏋️‍♀️ Training Configuration (Mobile-Net)
The MobileNet model is trained using the following configuration:

```python
# Train MobileNet model
def train_model(model, train_generator, valid_generator, filepath, epochs=10, batch_size=300):
    # Initial message to indicate the start of training
    print("Starting Model Training.")
    
    # Set up checkpointing to save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Define the number of training and validation samples
    n_train = 39117
    batch_size = 300
    n_valid = 9779

    try:
        # Start training the model with the specified parameters
        history = model.fit(
            train_generator,
            steps_per_epoch=n_train // batch_size,  # Total training steps per epoch
            epochs=epochs,  # Number of training epochs
            validation_data=valid_generator,
            validation_steps=n_valid // batch_size,  # Validation steps per epoch
            callbacks=callbacks_list  # List of callbacks (here, checkpointing)
        )
        print("Model Training Completed Successfully.")
    except Exception as e:
        # Error handling to catch any issues during training
        print(f"Error during training: {e}")
        return None

    # Extracting accuracy and loss values for training and validation to plot performance
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = range(1, len(acc) + 1)

    # Plot training and validation accuracy per epoch
    plt.plot(epoch_range, acc, 'b-', label='Training Accuracy')
    plt.plot(epoch_range, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()  # Add grid for easier readability
    plt.figure()

    # Plot training and validation loss per epoch
    plt.plot(epoch_range, loss, 'b-', label='Training Loss')
    plt.plot(epoch_range, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()  # Add grid for easier readability
    plt.show()

    # Return the trained model for further use
    return model
```
## Training Output

![download](https://github.com/user-attachments/assets/c230fccc-6e22-4a70-92a0-26b209aa4ab5)
![download](https://github.com/user-attachments/assets/418260e0-a269-45a0-af65-f85b02cd216e)


## 📊 Performance
The model is expected to achieve the following performance metrics on the test set:

```
[Loss, Accuracy]
[0.13525617122650146, 0.9586827754974365]

Accuracy: 95.87%
               precision    recall  f1-score   support

 rotated_left       0.97      0.94      0.95      1206
rotated_right       0.97      0.93      0.95      1241
      upright       0.95      0.98      0.96      1223
  upside_down       0.95      0.98      0.96      1219

     accuracy                           0.96      4889
    macro avg       0.96      0.96      0.96      4889
 weighted avg       0.96      0.96      0.96      4889
```


## 🏛️ VGG-16 Model Architecture
<img src="https://github.com/user-attachments/assets/ec8eabef-1398-4728-9ddd-e6d8ca76ffa7" 
     alt="Image" 
     width="60%" 
     height="50%">


```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt


# Function to create and compile the VGG16 model
def create_vgg16_model():
    # Define the input layer with a shape suitable for VGG16 (224x224 RGB image)
    input_layer = Input(shape=(150, 150, 3))

    # Load the VGG16 model, excluding the fully connected top layers (for feature extraction)
    # Pre-trained weights on ImageNet are used
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_layer)

    # Flatten the output from the last layer of VGG16 for input into fully connected layers
    x = Flatten()(vgg16.output)
    # Add a dense layer with 256 units and ReLU activation for intermediate processing
    x = Dense(256, activation='relu')(x)
    # Final output layer with 4 units and softmax activation for multiclass classification (4 classes)
    output_layer = Dense(4, activation='softmax')(x)

    # Define the complete model linking input and output layers
    model = Model(inputs=input_layer, outputs=output_layer)

    # Freeze all layers in the VGG16 base model to retain pre-trained features
    for layer in vgg16.layers:
        layer.trainable = False

    # Set up the Adam optimizer with a low learning rate for fine-tuning
    adam = Adam(learning_rate=1e-4)
    # Compile the model with categorical cross-entropy loss and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # Confirmation of successful model compilation and summary of model architecture
    print("Model Compiled Successfully.")
    print(model.summary())

    return model
```

## 🏋️‍♀️ Training Configuration (VGG-16)

```python
# Train VGG16 model
def train_vgg16_model(model, train_generator, valid_generator, filepath, epochs=10, batch_size=32):
    # Initial message to indicate the start of training
    print("Starting Model Training.")

    # Set up checkpointing to save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Define the number of training and validation samples
    n_train = 39117
    n_valid = 9779

    try:
        # Start training the model with the specified parameters
        history = model.fit(
            train_generator,
            steps_per_epoch=n_train // batch_size,  # Total training steps per epoch
            epochs=epochs,  # Number of training epochs
            validation_data=valid_generator,
            validation_steps=n_valid // batch_size,  # Validation steps per epoch
            callbacks=callbacks_list  # List of callbacks (here, checkpointing)
        )
        print("Model Training Completed Successfully.")
    except Exception as e:
        # Error handling to catch any issues during training
        print(f"Error during training: {e}")
        return None

    # Extracting accuracy and loss values for training and validation to plot performance
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = range(1, len(acc) + 1)

    # Plot training and validation accuracy per epoch
    plt.plot(epoch_range, acc, 'b-', label='Training Accuracy')
    plt.plot(epoch_range, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()  # Add grid for easier readability
    plt.figure()

    # Plot training and validation loss per epoch
    plt.plot(epoch_range, loss, 'b-', label='Training Loss')
    plt.plot(epoch_range, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()  # Add grid for easier readability
    plt.show()

    # Return the trained model for further use
    return model



```
## Training Output (Batch Size 32)

![download](https://github.com/user-attachments/assets/87c5350e-c82a-49df-b253-b7728a987bd4)
![download](https://github.com/user-attachments/assets/49241529-3819-4819-a2d2-9dccfee0e577)

## 📊 Performance (Batch Size 32)
The model is expected to achieve the following performance metrics on the test set:

```
[Loss, Accuracy]
[0.0863410159945488, 0.9713642597198486]
Accuracy: 97.14%
               precision    recall  f1-score   support

 rotated_left       0.96      0.98      0.97      1206
rotated_right       0.98      0.97      0.97      1241
      upright       0.98      0.97      0.98      1223
  upside_down       0.97      0.97      0.97      1219

     accuracy                           0.97      4889
    macro avg       0.97      0.97      0.97      4889
 weighted avg       0.97      0.97      0.97      4889

```

## Training Output (Batch Size 300)

![download](https://github.com/user-attachments/assets/8d822c43-1bf0-4115-a119-66073aa49378)
![download](https://github.com/user-attachments/assets/33b2566b-d7df-4436-8b91-b691a08a5175)

## 📊 Performance (Batch Size 300)
The model is expected to achieve the following performance metrics on the test set:

```
[Loss, Accuracy]
[0.09200621396303177, 0.9707506895065308]
Accuracy: 97.08%
               precision    recall  f1-score   support

 rotated_left       0.98      0.96      0.97      1206
rotated_right       0.97      0.96      0.97      1241
      upright       0.97      0.98      0.98      1223
  upside_down       0.96      0.98      0.97      1219

     accuracy                           0.97      4889
    macro avg       0.97      0.97      0.97      4889
 weighted avg       0.97      0.97      0.97      4889

```

## 🏛️ InceptionV3 Model Architecture

![image](https://github.com/user-attachments/assets/26e06446-661b-49a5-8e1a-8965d60790dc)


```python
def create_model_inception():
    input_layer = Input(shape=(150, 150, 3))
    inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_layer)

    x = Flatten()(inception.output)
    x = Dense(256, activation = "relu")(x)
    output_layer = Dense(4, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    sgd = SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    return model
```

## 🏋️‍♀️ Training Configuration (InceptionV3)

```python
def train(model, filepath):
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    n_train = 39117
    batch_size = 300
    n_valid = 9779

    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=n_train//batch_size,
            epochs=10,
            validation_data=valid_generator,
            validation_steps=n_valid//batch_size,
            callbacks=callbacks_list
        )
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    # Plot
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    return model
```
## Training Output

![download](https://github.com/user-attachments/assets/b442b726-c280-4ca7-9b2c-2220c0bf93c1)
![download](https://github.com/user-attachments/assets/069ec90b-fe95-4680-9a70-8b9dfc7ffbfa)

## 📊 Performance
The model is expected to achieve the following performance metrics on the test set:

```
[Loss, Accuracy]
[0.05644455924630165, 0.9803640842437744]

Accuracy: 98.04%
               precision    recall  f1-score   support

 rotated_left       0.98      0.98      0.98      1206
rotated_right       0.95      1.00      0.97      1241
      upright       1.00      0.98      0.99      1223
  upside_down       0.99      0.97      0.98      1219

     accuracy                           0.98      4889
    macro avg       0.98      0.98      0.98      4889
 weighted avg       0.98      0.98      0.98      4889
```

## 🔍 Predictions
The trained model can accurately predict the orientation of portrait images, as shown in the following example:

![1](https://github.com/user-attachments/assets/44acf762-f799-4b35-a93d-f756e37e0bca)

## Models That Can be Used

| **Model** | **Total Input Parameters** |
| --- | --- |
| MobileNet | 2,208,880 |
| MobileNetV2 | 2,259,696 |
| DenseNet121 | 7,038,952 |
| DenseNet169 | 12,753,032 |
| ResNet50 | 25,636,712 |
| InceptionV3 | 21,802,784 |
| DenseNet201 | 20,062,984 |
| ResNet101 | 44,707,176 |
| InceptionResNetV2 | 55,491,712 |
| ResNet152 | 60,192,584 |
| VGG16 | 138,357,544 |
| VGG19 | 143,667,240 |

## Usage
-----

### Import Models

```python
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, ResNet101, ResNet152,
    InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2,
    DenseNet121, DenseNet169, DenseNet201
)
```
## 📚 Articles and Research Papers
- [Orientation Visualization for Convolutional Neural Networks](https://www.cs.toronto.edu/~guerzhoy/oriviz/crv17.pdf)
- [Automatic Image Orientation Detection Using Deep Learning](https://www.tdcommons.org/cgi/viewcontent.cgi?article=6334&context=dpubs_series)
- [Going Deeper with Convolutions: The Inception Paper Explained](https://medium.com/aiguys/going-deeper-with-convolutions-the-inception-paper-explained-841a0c661fd3)
- [Unleashing the Power of MobileNet: A Comparison with Simple Convolutional Neural Networks](https://medium.com/@zaidbinmuzammil123/unleashing-the-power-of-mobilenet-a-comparison-with-simple-convolutional-neural-networks-71d49f8c86ef)
