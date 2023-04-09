import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import model_generation
from tensorflow import keras
from keras import layers
from keras.models import Sequential
model_list = list()

data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(model_generation.img_height,
                                           model_generation.img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ]
    )

model_list.append(Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(model_generation.img_height, model_generation.img_width, 3)),
        layers.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(64, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(128, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.282),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.282),
        layers.Dense(6, activation="softmax")
    ]))

model_list.append(Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(model_generation.img_height, model_generation.img_width, 3)),
        layers.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(64, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(128, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.286),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.286),
        layers.Dense(6, activation="softmax")
    ]))

model_list.append(Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(model_generation.img_height, model_generation.img_width, 3)),
        layers.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(64, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(128, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.287),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.287),
        layers.Dense(6, activation="softmax")
    ]))

model_list.append(Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(model_generation.img_height, model_generation.img_width, 3)),
        layers.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(64, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(128, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.283),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.283),
        layers.Dense(6, activation="softmax")
    ]))

model_list.append(Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(model_generation.img_height, model_generation.img_width, 3)),
        layers.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(64, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(128, (3,3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.284),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.284),
        layers.Dense(6, activation="softmax")
    ]))

datasets = model_generation.init_datasets()

for i, model in enumerate(model_list,start=1):
    print(f"RUN {i}/{len(model_list)}\n")
    model_generation.train(50, pre_model=model, datasets=datasets)

