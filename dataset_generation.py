import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
import random

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import Sequential

start_time = time.time()

img_height = 100
img_width = 100
batch_size = 32
random.seed(time.time())
seed = random.randint(0, 999999)

img_path = "img/Fruit_Flower_Veg"

# take 80% of the images and use them for training
train_ds = tf.keras.utils.image_dataset_from_directory(
    img_path,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# take 20% and use them for validation
val_ds = tf.keras.utils.image_dataset_from_directory(
    img_path,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

end_time = time.time()

print(f"{end_time - start_time} seconds elapsed")

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
epochs=1
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# plot some stuff based on the number of epochs
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

# import a random apple picture and try to predict it using the neural network
apple_path = "img/apple_example.jpg"

img = tf.keras.utils.load_img(
    apple_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)