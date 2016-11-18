import pickle, sys, os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

# $0 <data path> <output file>

# Params
batch_size = 32
nb_classes = 10
nb_epoch = 30
threshold = 0.98
do_weight = False
# 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'
optimizer = 'adam'
encode_layer = 5
encode_width = 512
new_item_size = 3000

all_label = pickle.load(open(sys.argv[1] + '/all_label.p', 'rb'))
all_unlabel = pickle.load(open(sys.argv[1] + '/all_unlabel.p', 'rb'))

# Create labeled
labeled = []
for num_class in all_label:
	for image in num_class:
		labeled.append(np.array(image))
labeled = np.array(labeled)
labeled = labeled.astype("float32")
labeled /= 255

# Create labeled answer
ans = []
index = 0
def set_answer(x):
	x[index] = 1
	return x
 
for i in range(10):
	index = i
	ans += map(set_answer, [[0 for i in range(10)] for i in range(500)])

ans = np.array(ans)

# Create unlabeled
unlabeled = []
for image in all_unlabel:
	unlabeled.append(np.array(image))
unlabeled = np.array(unlabeled)
unlabeled = unlabeled.astype("float32")
unlabeled /= 255

# Create encode NN
model = Sequential()
model.add(Dense(encode_width, activation='relu', input_shape=(3072, )))
for i in range(encode_layer) :
    model.add(Dense(encode_width, activation='relu'))
model.add(Dense(3072, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop', metrics=[ 'accuracy' ])
model.fit(labeled, labeled, batch_size=256, nb_epoch=200, verbose=1, validation_data=(labeled, labeled))
encoder = K.function([model.layers[0].input], [model.layers[(encode_layer + 1) / 2].output])

# Get the clustered unlabled data
encoded = encoder([labeled])[0]
before = [[0.0 for x in range(encode_width)] for x in range(10)]
for feature in range(encode_width):
	for category in range(10):
		for i in range(500):
			before[category][feature] += encoded[category * 500 + i][feature]
		before[category][feature] /= 500

after = encoder([unlabeled])[0]
cand_list = []
for image_num in range(45000):
	best_category = 0
	min = float('inf')
	for category in range(10):
		mse = 0.0
		for feature in range(encode_width):
			mse += (after[image_num][feature] - before[category][feature]) ** 2
		if mse < min:
			best_category = category
			min = mse
	cand_list.append([image_num, best_category, min])

cand_list.sort(key = lambda x: x[2])

new_data = []
new_ans = []
for i in range(new_item_size):
	new_data.append(unlabeled[cand_list[i][0]])
	index = cand_list[i][1]
	new_ans.append(set_answer([0 for x in range(10)]))

data = np.concatenate((labeled, new_data), axis=0).reshape(5000 + new_item_size, 3, 32, 32)
ans = np.concatenate((ans, new_ans), axis=0)

# Start normal training
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(data, ans, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)

try:
	os.remove(sys.argv[2])
except OSError:
	pass
model.save(sys.argv[2])
