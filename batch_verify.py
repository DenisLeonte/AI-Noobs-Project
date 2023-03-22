import tensorflow as tf
from dataset_generation import img_height,img_width
import numpy as np
import cv2

def verify(epochs, image_paths, correct_answers = [""], verbose = False):
	if len(image_paths) != len(correct_answers):
		raise Exception("Images and Answer have different lenght!")
	i=0
	images = []
	for image_path in image_paths:
		img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
		img_array = tf.keras.utils.img_to_array(img)
		img_array = tf.expand_dims(img_array, 0)
		images.append(img_array)
		correct_answers[i]=str(correct_answers[i]).split(" ")

	class_path = "img/Fruit_Flower_Veg/fruitname.txt"
	class_names = []
	with open(class_path, "r") as f:
		for line in f:
			class_names.append(line.removesuffix("\n"))

	model_list = []
	for i in range(1, epochs+1):
		model_list.append(tf.keras.models.load_model(f"models/checkpoints_RGB_1/model_epoch_{i:02d}.h5"))

	for j in range(0, len(image_paths)):
		print(f"\n{answers[j]}:\n")
		for i in range(0, epochs):
			predictions = model_list[i].predict(images[j], verbose=0)
			score = tf.nn.softmax(predictions[0])
			if verbose or correct_answers[j][0].lower() == class_names[np.argmax(score)].lower():
				print(
					"Epoch {:02d}: This image most likely belongs to {} with a {:.2f} percent confidence."
					.format(i+1,class_names[np.argmax(score)], 100 * np.max(score))
				)



img_paths = ["img/apple1.jpg","img/banana1.jpg","img/corn1.jpg","img/ex-image.jpg","img/grapes-purple.jpg",
						 "img/index.jpg","img/peach-1.jpg","img/pear-1.jpg","img/red_pepper1.jpg"]

answers = ["apple green", "banana 1", "corn ", "banana 2", "grapes purple", "apple red", "peach ", "pear ", "pepper red"]

verify(44, img_paths, answers)
