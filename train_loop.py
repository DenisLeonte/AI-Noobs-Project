import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import time
import random
from dataset_generation import train


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

for i in range(1, 11):
    train(train_ds, val_ds, i)