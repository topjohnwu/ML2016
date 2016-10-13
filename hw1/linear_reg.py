#!/usr/bin/python
import math
from operator import mul

# w0, w1, w2, w3, w4, w5, w6, w7, w8, b
params = [1.1] * 10
# The current gradient
grad = [0] * 10
# The square sum of all previous gradient
grad_sq_sum = [0] * 10
train = []
test = []
l_rate = 1.6
loop = 40000

def rate(i):
	return l_rate / math.sqrt(grad_sq_sum[i])

# w0x0 + w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6 + w7x7 + w8x8 + b
def estim(data):
	return sum(map(mul, params[0:9], data[0:9])) + params[9]
	
def gradient(data):
	estimate = estim(data)
	for i in range(len(data) - 1):
		grad[i] -= 2 * (data[9] - estimate) * data[i]
	grad[9] -= 2 * (data[9] - estimate)
	
def answer():
	temp = []
	for i in range(len(test)):
		temp.append([test[i][0], estim(test[i][1:])])
	return temp

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
		train.extend(map(lambda x: float(x), split[3:]))
infile.close()

# Read test data
infile = open('test_X.csv')
for line in infile:
	split = line.split(',')
	if (split[1] == "PM2.5"):
		temp = [split[0]]
		temp.extend(map(lambda x: float(x), split[2:]))
		test.append(temp)
infile.close()

# Training
for x in range(loop):
	for i in range(0, len(train) - 10):
		gradient(train[i : i + 10])
	for i in range(len(params)):
		grad_sq_sum[i] += math.pow(grad[i], 2)
		params[i] = params[i] - rate(i) * grad[i]
	grad = [0] * 10
	if x % 100 == 0:
		print(x, predict()) # Show progress
		print(params)

ans = answer()

str = 'id,value\n' + '\n'.join(map(lambda x: ','.join([x[0], str(x[1])]), ans))
outfile = open('submit.csv', 'w')
outfile.write(str)
outfile.close()
print(params)
