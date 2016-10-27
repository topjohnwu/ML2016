#!/usr/bin/python
import math
from operator import mul

train_data = []
train_ans = []

params = [0.0 for x in range(58)]
grad = [0.0 for x in range(58)]
grad_sq_sum = [0.0 for x in range(58)]

loop = 100
l_rate = 0.001

def estim(data):
	num = 0.0
	for i in range(len(data)):
		num += sum(map(mul, data, params[:len(data)]))
	num += params[len(params) - 1]
	return num

def sigmoid(num):
	try:
		return 1 / (1 + math.exp(0 - num))
	except OverflowError:
		return 0.0
	
def rate(i):
	return l_rate / math.sqrt(1 if grad_sq_sum[i] == 0 else grad_sq_sum[i])
	
def gradient(data, ans):
	estimate = sigmoid(estim(data))
	for i in range(len(data)):
		grad[i] -= (ans - estimate) * data[i]
	grad[len(grad) - 1] -= ans - estimate

def updateParam():
	global grad
	for i in range(len(params)):
		grad_sq_sum[i] += math.pow(grad[i], 2)
		params[i] -= rate(i) * grad[i]
	grad = [0.0 for x in range(58)]

def predict():
	num = 0.0
	for i in range(len(train_data)):
		num += (estim(train_data[i]) > 0) == train_ans[i]
	return num / len(train_data)

# Read training data
infile = open('spam_train.csv')
for line in infile:
	split = line.split(',')
	train_data.append(map(float, split[1:58]))
	train_ans.append(int(split[58]))
infile.close()

# Training
for x in range(loop):
	for i in range(len(train_data)):
		gradient(train_data[i], train_ans[i])
		updateParam()
	print(x + 1, predict()) # Show progress
	
# Read test and do estimation
result = 'id,label\n'
infile = open('spam_test.csv')
for line in infile:
	split = line.split(',')
	result += split[0] + ',' + str(int(estim(map(float, split[1:])) > 0))+ '\n'
infile.close()

# Output result
outfile = open('submit.csv', 'w')
outfile.write(result)
outfile.close()
