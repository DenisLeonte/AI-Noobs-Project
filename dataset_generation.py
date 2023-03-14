import tensorflow as tf
import numpy as np
import time
import random

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import Sequential

img_height = 100
img_width = 100
batch_size = 32

def train(train_ds, val_ds, epochs, plot=False):
    f = open('models/version.txt')
    ver_number = int(f.readline())
    f.close()
    f = open('models/version.txt', 'w')
    f.write(f"{ver_number+1}")
    f.close()

    class_names = train_ds.class_names
    #print(class_names)

    # Plot a bunch of pictures to visualize the data
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    #plt.show()

    # buffered prefetching, so you can yield data from disk without having I/O become blocking
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    num_classes = len(class_names)

    # build the neural network model
    model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
      layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
      layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
      layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
      layers.Flatten(),
      layers.Dense(1024, activation='relu'),
      layers.Dropout(0.2),
      layers.Dense(num_classes),
      layers.Dropout(0.2)
    ])

    # compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    # start training the neural network
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    # plot some stuff based on the number of epochs
    if plot:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

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

    model.save(f'models/model_{ver_number}.h5')

    return model