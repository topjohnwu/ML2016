#!/usr/bin/python
import math
import sys
from operator import mul

def estim(data):
	num = sum(map(mul, data, params[:len(data)]))
	num += params[len(params) - 1]
	return num

def sigmoid(num):
	try:
		return 1 / (1 + math.exp(0 - num))
	except OverflowError:
		return 0.0
	
def rate(grad_sum):
	return l_rate / math.sqrt(1 if grad_sum == 0 else grad_sum)
	
def gradient(data, ans):
	estimate = sigmoid(estim(data))
	for i in range(len(data)):
		grad[i] -= (ans - estimate) * data[i]
	grad[len(grad) - 1] -= ans - estimate
	return estimate

def updateParam():
	global grad
	for i in range(len(params)):
		grad_sq_sum[i] += math.pow(grad[i], 2)
		params[i] -= rate(grad_sq_sum[i]) * grad[i]
	grad = [0.0 for x in range(58)]

def predict(result):
	num = 0.0
	for i in range(len(train_ans)):
		num += ((result[i] > 0.5) == train_ans[i])
	return num / len(train_ans)
	
if sys.argv[1] == '--train':
	# Variables
	train_data = []
	train_ans = []

	params = [0.0 for x in range(58)]
	grad = [0.0 for x in range(58)]
	grad_sq_sum = [0.0 for x in range(58)]

	loop = int(sys.argv[2])
	l_rate = 0.05
	
	# Read training data
	infile = open(sys.argv[3])
	for line in infile:
		split = line.split(',')
		train_data.append(map(float, split[1:58]))
		train_ans.append(int(split[58]))
	infile.close()

	# Training
	for x in range(loop):
		result = []
		for i in range(len(train_data)):
			result.append(gradient(train_data[i], train_ans[i]))
			updateParam()
		# Dump parameter every 10 iterations
		if not((x + 1) % 10):
			print(x + 1, predict(result)) # Show progress
			string = ','.join(map(str, params))
			outfile = open(sys.argv[4], 'w')
			outfile.write(string)
	
	outfile.close()

elif sys.argv[1] == '--test':
	# Read parameters
	infile = open(sys.argv[2])
	params = map(float, infile.readline().split(','))
	infile.close()
	
	# Read test and do estimation
	result = 'id,label\n'
	infile = open(sys.argv[3])
	for line in infile:
		split = line.split(',')
		result += split[0] + ',' + str(int(estim(map(float, split[1:])) > 0))+ '\n'
	infile.close()

	# Output result
	outfile = open(sys.argv[4], 'w')
	outfile.write(result)
	outfile.close()
else:
	print(sys.argv[0] + ' --train: Train with training data')
	print(sys.argv[0] + ' --test: Output result from testing data')
