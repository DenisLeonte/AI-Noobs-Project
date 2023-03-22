import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import tensorboard as tb
from datetime import datetime
import numpy as np
import time
import random
import pandas as pd

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.callbacks import CSVLogger

img_height = 100
img_width = 100
batch_size = 64


def augment_image(img):
    img = tf.image.random_saturation(img, 0.9, 1.2)
    img = tf.image.random_hue(img, 0.02)
    return img


def checkpoint_name(epoch, logs):
    return "models/checkpoints/model_weights_epoch{}_valloss{:.4f}.h5".format(epoch, logs['val_loss'])

def network(num_classes):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ]
    )

    # build the neural network model
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

def train(train_ds, val_ds, epochs, plot=False):
    f = open('models/version.txt')
    ver_number = int(f.readline())
    f.close()
    f = open('models/version.txt', 'w')
    f.write(f"{ver_number+1}")
    f.close()

    class_names = train_ds.class_names
    #print(class_names)

    # buffered prefetching, so you can yield data from disk without having I/O become blocking
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #train_ds = train_ds.map(augment_image)
    #val_ds = val_ds.map(augment_image)

    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    num_classes = len(class_names)

    model = network(num_classes)

    # compile the model
    model.compile(optimizer='adam',
                  loss=tf.metrics.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()

    model_checkpoints = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/checkpoints_RGB_1/model_epoch_{epoch:02d}.h5",
        monitor='val_loss',
        save_freq='epoch',
        verbose=0
    )

    log_dir = "boards/"+ datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback=keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)

    # start training the neural network
    csv_logger = CSVLogger('training.log', separator=',', append=False)
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      callbacks=[model_checkpoints, csv_logger, tensorboard_callback]
    )

    # plot some data
    if plot:
        plot_data(epochs)

    model.save(f'models/model_{ver_number}.h5')

    return model

def plot_data(epochs):
    log_data = pd.read_csv('training.log', sep=',', engine='python')

    acc = log_data['accuracy']
    val_acc = log_data['val_accuracy']

    loss = log_data['loss']
    val_loss = log_data['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def resume_training(train_ds, val_ds, epochs, last_epoch):
    checkpoint_dir = "models/checkpoints_RGB_1"
    log_dir="boards/"

    model = keras.models.load_model(f"models/checkpoints_RGB_1/model_epoch_{last_epoch:02d}.h5")

    csv_logger = CSVLogger('training.log', separator=',', append=True)
    tensorboard_callback=keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)

    model_checkpoints = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/checkpoints_RGB_1/model_epoch_{epoch:02d}.h5",
        monitor='val_loss',
        save_freq='epoch',
        verbose=0
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=last_epoch,
        callbacks=[model_checkpoints, csv_logger, tensorboard_callback]
    )

plot_data(54)