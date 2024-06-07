# Image-Based Forgery Detection

This project aims to detect image forgeries using various deep learning models. The dataset consists of 2000 images, with 1000 authentic and 1000 forged images. The project employs models such as Basic CNN, VGG16, InceptionV3, ResNet50, and MobileNetV2 to classify the images into authentic and forged categories.

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/image-forgery-detection.git
    cd image-forgery-detection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

Ensure your dataset is structured as follows:

dataset/authentic/image1.jpg,
dataset/authentic/image2.jpg ...

dataset/forged/image1.jpg
dataset/forged/image2.jpg ...

You must replace the path with the path to your dataset.

Dataset link: [Image Forgery Dataset](https://www.kaggle.com/datasets/sophatvathana/casia-dataset/data)


## Model Training and Evaluation

### Error Level Analysis (ELA)

The project uses Error Level Analysis (ELA) to preprocess images, highlighting differences between the original and compressed images, which helps in detecting manipulations.

### Basic CNN Model

The Basic CNN model consists of two Conv2D layers followed by MaxPool2D, Dropout, Flatten, and Dense layers.

### VGG16 Model

VGG16 is used with pre-trained weights and additional Dense layers to adapt to our classification task. The model's base layers are frozen during training.

### InceptionV3 Model

InceptionV3, known for its multi-scale architecture, is used with frozen pre-trained weights and additional Dense layers.

### ResNet50 Model

ResNet50, which employs residual connections, is used with frozen pre-trained weights and additional Dense layers.

### MobileNetV2 Model

MobileNetV2, designed for mobile and embedded vision applications, is used with frozen pre-trained weights and additional Dense layers.

### Training and Evaluation

All models are trained and evaluated using the same training and validation splits. The best models are saved during training.

```python
# Example code snippet for training and evaluating a model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=32, validation_data=(X_val, Y_val), callbacks=[checkpoint_callback])
```
### Results
The performance of each model is evaluated using accuracy, confusion matrix, and classification report. Below is an example of how to print and plot these metrics.

### Example code snippet for evaluating model performance
```python
Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
print("Classification report:")
print(classification_report(Y_true, Y_pred_classes, target_names=['forged', 'authentic']))
print("Overall accuracy: {:.2f}%".format(accuracy_score(Y_true, Y_pred_classes) * 100))
```
