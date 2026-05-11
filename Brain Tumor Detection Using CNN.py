# Brain Tumor Detection Using CNN



# ==========================================
# Brain Tumor Detection using CNN
# ==========================================

# Import required libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from os import listdir
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Flatten,
    Dense,
    ZeroPadding2D
)

# ==========================================
# Dataset Path
# ==========================================

image_dir = 'dataset/'

# ==========================================
# Data Augmentation
# ==========================================

augmented_data_path = 'augmented-data/'

os.makedirs(augmented_data_path, exist_ok=True)
os.makedirs(augmented_data_path + 'yes/', exist_ok=True)
os.makedirs(augmented_data_path + 'no/', exist_ok=True)


def augment_data(file_dir, n_generated_samples, save_to_dir):
    """
    Generate augmented images to increase dataset size.
    """

    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=(0.3, 1.0),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    for filename in listdir(file_dir):
        image = cv2.imread(file_dir + '/' + filename)
        image = image.reshape((1,) + image.shape)

        save_prefix = 'aug'

        i = 0
        for batch in data_gen.flow(
            image,
            batch_size=1,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format='jpg'
        ):
            i += 1
            if i > n_generated_samples:
                break


# Generate augmented images
augment_data(file_dir=image_dir + 'yes', n_generated_samples=6,
             save_to_dir=augmented_data_path + 'yes')

augment_data(file_dir=image_dir + 'no', n_generated_samples=9,
             save_to_dir=augmented_data_path + 'no')


# ==========================================
# Image Preprocessing
# ==========================================


def crop_brain_contour(image, plot=False):
    """
    Crop the brain region from MRI image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Image')

        plt.show()

    return new_image


# ==========================================
# Load Dataset
# ==========================================


def load_data(dir_list, image_size):
    """
    Load images and labels.
    """

    X = []
    y = []

    image_width, image_height = image_size

    for directory in dir_list:

        for filename in listdir(directory):
            image = cv2.imread(directory + '/' + filename)

            image = crop_brain_contour(image)
            image = cv2.resize(image, dsize=(image_width, image_height),
                               interpolation=cv2.INTER_CUBIC)

            image = image / 255.0

            X.append(image)

            if directory.endswith('yes'):
                y.append([1])
            else:
                y.append([0])

    X = np.array(X)
    y = np.array(y)

    return X, y


IMG_WIDTH, IMG_HEIGHT = (240, 240)

X, y = load_data(
    [augmented_data_path + 'yes', augmented_data_path + 'no'],
    (IMG_WIDTH, IMG_HEIGHT)
)


# ==========================================
# Split Dataset
# ==========================================

X_train, X_test_val, y_train, y_test_val = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

X_test, X_val, y_test, y_val = train_test_split(
    X_test_val,
    y_test_val,
    test_size=0.5,
    random_state=42
)

print('Training examples:', X_train.shape[0])
print('Validation examples:', X_val.shape[0])
print('Testing examples:', X_test.shape[0])


# ==========================================
# Build CNN Model
# ==========================================


def build_model(input_shape):
    """
    Build CNN architecture.
    """

    X_input = Input(input_shape)

    X = ZeroPadding2D((2, 2))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((4, 4))(X)
    X = MaxPooling2D((4, 4))(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=X)

    return model


IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
model = build_model(IMG_SHAPE)

model.summary()


# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ==========================================
# Train Model
# ==========================================

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_val, y_val)
)


# ==========================================
# Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Accuracy: {accuracy * 100:.2f}%')


# ==========================================
# Save Model
# ==========================================

model.save('brain_tumor_model.h5')

print('Model saved successfully!')


# ==========================================
# Plot Accuracy and Loss
# ==========================================


def plot_metrics(history):

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()


plot_metrics(history)
```



