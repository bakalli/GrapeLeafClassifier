import tensorflow as tf 
from tensorflow import keras 

# Helper libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc 
from PIL import Image 
from random import shuffle 
import numpy

print(tf.__version__)


def _parse_function(filename,label):
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string)
	image_resized = tf.image.resize_images(image_decoded, [256, 256])
	return image_resized, label

#LOAD FILENAMES
def load_data():


	IMG_SIZE_ALEXNET = 227


	diseased_filenames = []
	healthy_filenames = []
	with open("healthy.txt",'r') as ht:
		healthy_filenames = ht.readlines()
	with open("black_rot.txt",'r') as uht:
		diseased_filenames = diseased_filenames + uht.readlines()
	with open("leaf_blight.txt",'r') as uht:
		diseased_filenames = diseased_filenames + uht.readlines()
	with open("measles.txt",'r') as uht:
		diseased_filenames = diseased_filenames + uht.readlines()



	filenames =  ([["unhealthy/"+d.strip(),1] for d in diseased_filenames] + [["healthy/"+d.strip(),0] for d in healthy_filenames])
	shuffle(filenames)
	all_data = [numpy.array([numpy.array(Image.open(d[0])),d[1]]) for d in filenames]
 
	end_index = (len(all_data) // 6 ) * 5

	train_data = all_data[:end_index]
	test_data = all_data[end_index+1:]

	for i in range(len(train_data)):
		train_data[i][0] = cv2.resize(train_data[i][0],(IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET))

	for i in range(len(test_data)):
		test_data[i][0] = cv2.resize(test_data[i][0],(IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET))

	train_end_index = (len(train_data) // 5)*4

	train = train_data[:train_end_index]
	cv = train_data[train_end_index:]

	X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
	Y = np.array([i[1] for i in train])


	cv_x = np.array([i[0] for i in cv]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
	cv_y = np.array([i[1] for i in cv])
	test_x = np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
	test_y = np.array([i[1] for i in test_data])

	print(X.shape)

	print(Y.shape)

	print(cv_x.shape)

	print(test_x.shape)



# def train():

#   train_data, test_data = load_data()

#   model = keras.Sequential([
#       keras.layers.Flatten(input_shape=(256, 256)),
#       keras.layers.Dense(128, activation=tf.nn.relu),
#       keras.layers.Dense(2, activation=tf.nn.softmax)
#   ])

#   cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))
#   optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

#   #Initializing weights
#   init = tf.global_variables_initializer()

#   model.compile(optimizer=optimizer, 
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])


#   with tf.Session() as sess:
#       sess.run(init)
#   num_epochs = 10 
#   train_loss_results = []
#   train_accuracy_results = []

#   for epoch in range(num_epochs):
#   ##TRAINING STEP
#   #model.fit(..training data..)...


if __name__ == "__main__":
	load_data()

