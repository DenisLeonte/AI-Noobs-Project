import tensorflow as tf
from dataset_generation import img_height,img_width
import numpy as np

def verify(epochs, image_path, correct_answer = ""):
	found = False
	class_path = "img/Fruit_Flower_Veg/fruitname.txt"
	class_names = []
	with open(class_path, "r") as f:
		for line in f:
			class_names.append(line.removesuffix("\n"))

	model_list = []
	for i in range(1, epochs+1):
		model_list.append(tf.keras.models.load_model(f"models/checkpoints/model_epoch_{i:02d}.h5"))
	img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)
	for i in range(0, epochs):
		predictions = model_list[i].predict(img_array, verbose=0)
		score = tf.nn.softmax(predictions[0])
		if correct_answer == "" or class_names[np.argmax(score)] == correct_answer:
			print(
				"Epoch {:02d}: This image most likely belongs to {} with a {:.2f} percent confidence."
				.format(i+1,class_names[np.argmax(score)], 100 * np.max(score))
			)
			found = True
	if found == False:
		print("No models were correct!")

verify(4, "img/red_pepper1.jpg")