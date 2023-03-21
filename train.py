import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from dataset_generation import img_height, img_width, batch_size, train
import tensorflow as tf
import numpy as np
import time
import random

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import Sequential

img_path = "img/Fruit_Flower_Veg"
random.seed(time.time())
seed = random.randint(0, 999999)

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


epochs = 500

#uncomment this when you want to train the AI
model = train(train_ds, val_ds, epochs)

#comment this when you want to train the AI
#model = tf.keras.models.load_model("models/model_3.h5")

class_names = train_ds.class_names

# import a random apple picture and try to predict it using the neural network
apple_path = "img/index.jpg"


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