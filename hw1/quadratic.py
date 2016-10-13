#!/usr/bin/python
import math
from operator import mul

params_1 = [1] * 9
params_2 = [1] * 9
b = 1
# The current gradient
grad_1 = [0] * 9
grad_2 = [0] * 9
grad_b = 0
# The square sum of all previous gradient
grad_sq_sum_1 = [0] * 9
grad_sq_sum_2 = [0] * 9
grad_sq_sum_b = 0

train = []
test = []
test_name = []

l_rate = 1
loop = 37000
best_loss = 1000000

def estim(data):
	num = 0
	num += sum(map(mul, params_1[0:9], data[0:9]))
	num += sum(map(lambda x, y: x * math.pow(y, 2), params_2[0:9], data[0:9]))
	num += b
	return num
	
def gradient(data):
	global grad_b
	estimate = estim(data)
	for i in range(len(params_1)):
		grad_1[i] -= 2 * (data[9] - estimate) * data[i]
	for i in range(len(params_2)):
		grad_2[i] -= 2 * (data[9] - estimate) * math.pow(data[i], 2)
	grad_b -= 2 * (data[9] - estimate)
	
def updateParam():
	global grad_1, grad_2, grad_b, grad_sq_sum_b, b
	for i in range(len(grad_1)):
		grad_sq_sum_1[i] += math.pow(grad_1[i], 2)
		params_1[i] -= l_rate / math.sqrt(grad_sq_sum_1[i]) * grad_1[i]
	for i in range(len(grad_2)):
		grad_sq_sum_2[i] += math.pow(grad_2[i], 2)
		params_2[i] -= l_rate / math.sqrt(grad_sq_sum_2[i]) * grad_2[i]
	grad_sq_sum_b += math.pow(grad_b, 2)
	b -= l_rate / math.sqrt(grad_sq_sum_b) * grad_b
	grad_1 = [0] * 9
	grad_2 = [0] * 9
	grad_b = 0
	
def predict():
	temp = []
	for i in range(len(train) - 10):
		temp.append(train[i + 9] - estim(train[i : i + 9]))
	return math.sqrt(sum(map(lambda x: math.pow(x, 2), temp)) / len(temp))

# Read training data
infile = open('train.csv')
for line in infile:
	split = line.split(',')
	if (split[2] == "PM2.5"):
		train.extend(map(float, split[3:]))
infile.close()

# Read test data
infile = open('test_X.csv')
for line in infile:
	split = line.split(',')
	if (split[1] == "PM2.5"):
		test_name.append(split[0])
		test.append(map(float, split[2:]))
infile.close()

# Training
for x in range(loop):
	for i in range(0, len(train) - 10):
		gradient(train[i : i + 10])
	updateParam()
	if x % 100 == 0:
		print(x, predict()) # Show progress
		print(params_1)
		print(params_2)
		print(b)

# Output
ans = map(lambda x: str(estim(x)), test)
lines = map(lambda x, y: ','.join([x, y]), test_name, ans)
result = 'id,value\n' + '\n'.join(lines)
outfile = open('submit.csv', 'w')
outfile.write(result)
outfile.close()
