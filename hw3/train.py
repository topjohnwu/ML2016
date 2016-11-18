import pickle, sys, os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

# $0 <data path> <output file>

# Params
batch_size = 32
nb_classes = 10
nb_epoch = 30
threshold = 0.98
do_weight = True
# 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'
optimizer = 'adam'

all_label = pickle.load(open(sys.argv[1] + '/all_label.p', 'rb'))
all_unlabel = pickle.load(open(sys.argv[1] + '/all_unlabel.p', 'rb'))

# Create labeled
labeled = []
for num_class in all_label:
	for image in num_class:
		labeled.append(np.array(image).reshape(3, 32, 32))
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
	unlabeled.append(np.array(image).reshape(3, 32, 32))
unlabeled = np.array(unlabeled)
unlabeled = unlabeled.astype("float32")
unlabeled /= 255

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=labeled.shape[1:]))
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

model.fit(labeled, ans, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)

# Model is supervised trained, now predict unlabeled data
# Dump the supervised model
try:
	os.remove('supervised.model')
except OSError:
	pass
model.save('supervised.model')
predicts = model.predict(unlabeled, verbose=1)
# sampled[n] = [data, answer, weight]
sampled = []

for i in range(len(predicts)):
	predict = predicts[i]
	max = predict[0]
	assumption = 0
	for j in range(len(predict)):
		if predict[j] > max:
			assumption = j
			max = predict[j]
	if max > threshold:
		index = assumption
		sampled.append([unlabeled[i], set_answer([0 for i in range(10)]), max])

data = np.concatenate((labeled, map(lambda x: x[0], sampled)), axis=0)
ans = np.concatenate((ans, map(lambda x: x[1], sampled)), axis=0)
weights = np.array([1.0 for x in range(5000)] + map(lambda x: x[2], sampled)) if do_weight else None

print(data.shape, ans.shape)	

model.fit(data, ans, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, sample_weight=weights)

try:
	os.remove(sys.argv[2])
except OSError:
	pass
model.save(sys.argv[2])

