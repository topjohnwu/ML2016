#!/usr/bin/python
import math
from operator import mul

# Initialize values
params = [[0 for x in range(9)] for x in range(18)] + [[0]]
grad = [[0 for x in range(9)] for x in range(18)] + [[0]]
grad_sq_sum = [[0 for x in range(9)] for x in range(18)] + [[0]]
train = [None] * 18
test = []
test_name = []
leak = []

loop = 40000
l_rate = 1
lam = 0
prev = 0
dataType = {
	'AMB_TEMP': [True, 0],
	'CH4': [True, 1],
	'CO': [True, 2],
	'NMHC': [True, 3],
	'NO': [True, 4],
	'NO2': [True, 5],
	'NOx': [True, 6],
	'O3': [True, 7],
	'PM10': [True, 8],
	'PM2.5': [True, 9],
	"RAINFALL": [False, 10],
	"RH": [True, 11],
	'SO2': [True, 12],
	"THC": [True, 13], 
	"WD_HR": [True, 14],
	"WIND_DIREC": [False, 15],
	"WIND_SPEED": [True, 16],
	"WS_HR": [True, 17],
}
def include(string):
	return dataType.get(string, [False])[0]

def getIndex(string):
	return dataType.get(string)[1]

def estim(data):
	num = 0
	for i in range(len(data)):
		if data[i] != None:
			num += sum(map(mul, params[i][prev:9], data[i][prev:9]))
	num += params[len(params) - 1][0]
	return num
	
def gradient(data):
	actual = data[getIndex('PM2.5')][9]
	estimate = estim(data)
	for i in range(len(data)):
		if data[i] != None:
			for j in range(prev, len(grad[i])):
				grad[i][j] -= 2 * (actual - estimate) * data[i][j]
				grad[i][j] += 2 * lam * params[i][j]
	grad[len(grad) - 1][0] -= 2 * (actual - estimate)

def updateParam():
	global grad
	size = len(train)
	for i in range(size):
		if train[i] != None:
			for j in range(prev, len(params[i])):
				grad_sq_sum[i][j] += math.pow(grad[i][j], 2)
				params[i][j] -= rate(i, j) * grad[i][j]
	grad_sq_sum[size][0] += math.pow(grad[size][0], 2)
	params[size][0] -= rate(size, 0) * grad[size][0]
	grad = [[0 for x in range(9)] for x in range(18)] + [[0]]
	
def predict():
	temp = []
	index = getIndex('PM2.5')
	for i in range(len(train[index]) - 10):
		temp.append(train[index][i + 9] - estim(map(lambda x: x[i : i + 10] if x != None else None, train)))
	return math.sqrt(sum(map(lambda x: math.pow(x, 2), temp)) / len(temp))

def rate(i, j):
	return l_rate / math.sqrt(grad_sq_sum[i][j])

# Read training data
infile = open('train.csv')
for line in infile:
	split = line.split(',')
	if include(split[2]):
		if train[getIndex(split[2])] == None:
			train[getIndex(split[2])] = map(lambda x: 0 if 'NR' in x else float(x), split[3:])
		else:
			train[getIndex(split[2])].extend(map(lambda x: 0 if 'NR' in x else float(x), split[3:]))
infile.close()

# Read test data
infile = open('test_X.csv')
curr = None
day = [None] * 18
for line in infile:
	split = line.split(',')
	if curr == None:
		curr = split[0]
	
	if (split[0] != curr):
		test.append(day)
		test_name.append(curr)
		curr = split[0]
		# Reset day
		day = [None] * 18
		
	if include(split[1]):
		day[getIndex(split[1])] = map(lambda x: 0 if 'NR' in x else float(x), split[2:])

infile.close()
test_name.append(curr)
test.append(day)

# Training
for x in range(loop):
	for i in range(len(train[getIndex('PM2.5')]) - 10):
		gradient(map(lambda x: x[i : i + 10] if x != None else None, train))
	updateParam()
	if x % 100 == 0:
		print(x, predict()) # Show progress
		print(params)
		
# Output result
ans = map(lambda x: str(estim(x)), test)
lines = map(lambda x, y: ','.join([x, y]), test_name, ans)
result = 'id,value\n' + '\n'.join(lines)
outfile = open('submit.csv', 'w')
outfile.write(result)
outfile.close()
	