import sys

def getKey(item):
	return float(item)

arr = []
infile = open(sys.argv[2])
output = open('ans1.txt', 'w')

for line in infile:
	arr.append(line.split()[int(sys.argv[1])])

str = ','.join(sorted(arr, key=getKey))

output.write(str)

infile.close()
output.close()