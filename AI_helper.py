import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class AI_Helper:
    def __init__(self):
        model_path = "models/used model/active_model.h5"
        self.model = keras.models.load_model(model_path, compile=False)
        class_path = "img/Fruit_Flower_Veg/fruitname.txt"
        self.class_names = []
        with open(class_path, "r") as f:
            for line in f:
                self.class_names.append(line.removesuffix("\n"))

    def predict_image(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=(100, 100))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        return "This image most likely belongs to {} with a {:.2f} percent confidence." \
            .format(self.class_names[np.argmax(score)], 100 * np.max(score))
