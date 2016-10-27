#!/usr/bin/python
import math
import random
import sys
from operator import mul

def estim(data, params):
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
	
def rate(grad_sum):
	return l_rate / math.sqrt(1 if grad_sum == 0 else grad_sum)
	
def gradient(data, ans, layer):
	if layer == len(net_params):
		return data[0]
	result = []
	for node in range(len(net_params[layer])):
		estimate = sigmoid(estim(data, net_params[layer][node]))
		result.append(estimate)
		# Update gradient list
		for i in range(len(data)):
			grad[layer][node][i] -= (ans - estimate) * data[i]
		grad[layer][node][len(grad[layer][node]) - 1] -= ans - estimate
	return gradient(map(lambda x: x - 0.5, result), ans, layer + 1)

def updateParam():
	global grad
	for layer in range(len(net_params)):
		for node in range(len(net_params[layer])):
			for i in range(len(net_params[layer][node])):
				grad_sq_sum[layer][node][i] += math.pow(grad[layer][node][i], 2)
				net_params[layer][node][i] -= rate(grad_sq_sum[layer][node][i]) * grad[layer][node][i]
	grad = [[[0.0 for x in range(58)] for x in range(57)] for x in range(2)]

def predict(results):
	num = 0.0
	for i in range(len(results)):
		num += (results[i] > 0) == train_ans[i]
	return num / len(results)
	
def getResult(data, layer):
	if layer == len(net_params):
		return data[0]
	result = []
	for node in range(len(net_params[layer])):
		estimate = sigmoid(estim(data, net_params[layer][node]))
		result.append(estimate)
	return getResult(map(lambda x: x - 0.5, result), layer + 1)

if sys.argv[1] == '--train':
	# Variables
	train_data = []
	train_ans = []

	# Initialize data
	net_params = [[[random.uniform(0, 0.01) for x in range(58)] for x in range(57)], [[random.uniform(0, 0.02) for x in range(58)] for x in range(1)]]
	grad = [[[0.0 for x in range(58)] for x in range(57)], [[0.0 for x in range(58)] for x in range(1)]]
	grad_sq_sum = [[[0.0 for x in range(58)] for x in range(57)], [[0.0 for x in range(58)] for x in range(1)]]

	loop = int(sys.argv[2])
	l_rate = 0.001
	
	# Read training data
	infile = open(sys.argv[3])
	for line in infile:
		split = line.split(',')
		train_data.append(map(float, split[1:58]))
		train_ans.append(int(split[58]))
	infile.close()
	
	outfile = open(sys.argv[4], 'w')
	
	# Training
	for x in range(loop):
		results = []
		for i in range(len(train_data)):
			results.append(gradient(train_data[i], train_ans[i], 0))
			updateParam()
		outfile.seek(0)
		string = '\n'.join(map(lambda layer: '|'.join(map(lambda node: ','.join(map(str, node)), layer)), net_params))
		outfile.write(string)
		print(x + 1, predict(results)) # Show progress
	
	outfile.close()

elif sys.argv[1] == '--test':
	# Read parameters
	infile = open(sys.argv[2])
	net_params = []
	for layer in infile:
		net_params.append(map(lambda node: map(float, node.split(',')), layer.split('|')))
	infile.close()
	
	# Read test and do estimation
	result = 'id,label\n'
	infile = open(sys.argv[3])
	for line in infile:
		split = line.split(',')
		result += split[0] + ',' + str(int(getResult(map(float, split[1:]), 0) > 0))+ '\n'
	infile.close()

	# Output result
	outfile = open(sys.argv[4], 'w')
	outfile.write(result)
	outfile.close()
else:
	print(sys.argv[0] + ' --train: Train with training data')
	print(sys.argv[0] + ' --test: Output result from testing data')

