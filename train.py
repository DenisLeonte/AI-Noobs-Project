import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from dataset_generation import img_height, img_width, batch_size, train, resume_training
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
def init_datasets():
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
    return (train_ds, val_ds)

def train(epochs):

    #uncomment this when you want to train the AI
    model = train(train_ds, val_ds, epochs)

def res_train(epochs, last_epoch):
    resume_training(train_ds, val_ds,epochs, last_epoch)


#res_train(500,7)