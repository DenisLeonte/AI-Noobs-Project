import keras
import tensorflow as tf
from train import img_width,img_height
import numpy as np
import matplotlib.pyplot as plt


model = 0
class_names = []
def init_AI_helper():
	model_path = "models/used model/active_model.h5"
	model = keras.models.load_model(model_path)
	class_path = "img/Fruit_Flower_Veg/fruitname.txt"
	class_names = []
	with open(class_path, "r") as f:
		for line in f:
			class_names.append(line.removesuffix("\n"))
def predict_image(image_path):

	img = plt.imread(image_path)
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)

	predictions = model.predict(img_array, verbose=0)
	score = tf.nn.softmax(predictions[0])
	return "This image most likely belongs to {} with a {:.2f} percent confidence."\
		.format(class_names[np.argmax(score)], 100 * np.max(score))