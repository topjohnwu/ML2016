import pickle, sys
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

all_label = pickle.load(open('all_label.p', 'rb'))

# Create labeled
labeled = []
for num_class in all_label:
	for image in num_class:
		labeled.append(np.array(image).reshape(3, 32, 32))
labeled = np.array(labeled)

# Create answer
ans = []
index = 0
def assign(x):
  x[index] = 1
  return x
 
for i in range(10):
  index = i
  ans += map(assign, [[0 for i in range(10)] for i in range(500)])

ans = np.array(ans)

batch_size = 32
nb_classes = 10
nb_epoch = 200

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

# let's train the model using SGD + momentum (how original).
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sys.argv[1], metrics=['accuracy'])

labeled = labeled.astype("float32")
labeled /= 255

model.fit(labeled, ans, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)

model.save(sys.argv[2])

