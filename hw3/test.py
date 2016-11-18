import pickle, sys
import numpy as np

# $0 <data folder> <input model> <output csv>

from keras.models import load_model

test_data = pickle.load(open(sys.argv[1] + '/test.p', 'rb'))

test = []
for image in test_data['data']:
    test.append(np.array(image).reshape(3, 32, 32))

test = np.array(test)

test = test.astype('float32')
test /= 255

model = load_model(sys.argv[2])
predicts = model.predict(test, verbose=1)

final = []

for i in range(len(predicts)):
    predict = predicts[i]
    max = predict[0]
    assumption = 0
    for j in range(len(predict)):
        if predict[j] > max:
            assumption = j
            max = predict[j]
    final.append((i, assumption))

output = open(sys.argv[3], 'w')
output.write('ID,class\n' + '\n'.join(map(lambda x: ','.join(map(str, x)), final)))
output.close
