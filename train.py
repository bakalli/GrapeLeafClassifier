import tensorflow as tf 
from tensorflow import keras 

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc 

print(tf.__version__)


def _parse_function(filename,label):
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string)
	image_resized = tf.image.resize_images(image_decoded, [28, 28])
	return image_resized, label

#LOAD FILENAMES
def load_data():
	diseased_filenames = []
	healthy_filenames = []
	with open("healthy.txt",'r') as ht:
		healthy_filenames = tf.constant(ht.readlines())
	with open("black_rot.txt",'r') as uht:
		diseased_filenames = diseased_filenames + uht.readlines()
	with open("leaf_blight.txt",'r') as uht:
		diseased_filenames = diseased_filenames + uht.readlines()
	with open("measles.txt",'r') as uht:
		diseased_filenames = diseased_filenames + uht.readlines()
	diseased_filenames = tf.constant(diseased_filenames)



	labels = tf.constant([1 for d in diseased_filenames] + [0 for h in healthy_filenames])

	##TODO: NEED TO SEPARATE DATA INTO TRAINING AND TESTING
	##TODO: NEED TO PREPROCESS DATA (maybe div by 255)

	dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	dataset = dataset.map(_parse_function)
	batched_dataset = dataset.batch(4)

def train():
	model = keras.Sequential([
	    keras.layers.Flatten(input_shape=(28, 28)),
	    keras.layers.Dense(128, activation=tf.nn.relu),
	    keras.layers.Dense(2, activation=tf.nn.softmax)
	])



	model.compile(optimizer=tf.train.AdamOptimizer(), 
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	##TRAINING STEP
	#model.fit(..training data..)...


if __name__ = "__main__":
	main()

