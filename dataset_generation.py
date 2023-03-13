import tensorflow as tf
import time
import random

start_time=time.time()

img_height=100
img_width=100
batch_size=32
random.seed(time.time())
seed=random.randint(0,999999)

img_path = "img/Fruit_Flower_Veg"

train_ds=tf.keras.utils.image_dataset_from_directory(
	img_path,
	validation_split=0.2,
	subset="training",
	seed = seed,
	image_size=(img_height,img_width),
	batch_size=batch_size
)

val_ds=tf.keras.utils.image_dataset_from_directory(
	img_path,
	validation_split=0.2,
	subset="validation",
	seed=seed,
	image_size=(img_height,img_width),
	batch_size=batch_size
)

end_time=time.time()

print(f"{end_time-start_time} seconds elapsed")