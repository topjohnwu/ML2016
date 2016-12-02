import numpy as np
import sys
import string

from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import matplotlib.colors as colors

data_path = sys.argv[1]
output_path = sys.argv[2]
stop_words = stopwords.words('english')

# Tf-Idf settings
max_df = 0.4
min_df = 2

# lsa settings
n_components = 20

# k-means settings
n_clusters = 70
n_init = 100
max_iter = 100

def preprocess(source, target):
	for line in source:
		line = line.lower()
		for punc in string.punctuation:
			line = line.replace(punc, ' ')
		line.replace('\n', '')
		split = line.split()
		line = ' '.join([ x for x in split if x not in stop_words ])
		target.append(line)

print('Loading titles')
title_file = open(data_path + "/title_StackOverflow.txt", 'r')
titles = []
preprocess(title_file, titles)
title_file.close()

print('loading docs')
doc_file = open(data_path + "/docs.txt", 'r')
docs = []
preprocess(doc_file, docs)
doc_file.close()

print("Extracting features from the training dataset using a sparse vectorizer")
vectorizer = TfidfVectorizer(input = docs, max_df = max_df, min_df = min_df, stop_words = 'english', use_idf = True)
print()
X = vectorizer.fit_transform(titles)
print("n_samples: %d, n_features: %d" % X.shape)

if n_components:
	print("Performing dimensionality reduction using LSA")
	svd = TruncatedSVD(n_components)
	normalizer = Normalizer(copy = False)
	lsa = make_pipeline(svd, normalizer)

	X = lsa.fit_transform(X)

	explained_varience = svd.explained_variance_ratio_.sum()
	print("Explained varience of the SVD step: {}%".format(int(explained_varience * 100)))
	print()

kmean = KMeans(n_clusters = n_clusters, init = 'k-means++', n_jobs = -1,
			max_iter = max_iter, n_init = n_init, verbose = True)

print("Clustering sparse data with %s" % kmean)
kmean.fit(X)
print()

print("Top terms per cluster:")
if n_components:
	original_space_centroids = svd.inverse_transform(kmean.cluster_centers_)
	order_centroids = original_space_centroids.argsort()[:, ::-1]
else:
	order_centroids = kmean.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(n_clusters):
	print("Cluster %d:" % i, end = '')
	for ind in order_centroids[i, :10]:
		print(' %s' % terms[ind], end = '')
	print()

clusters = np.array(kmean.labels_)

print('Loading checks')
check_file = open(data_path + "/check_index.csv", 'r')
checks = []
for i, line in enumerate(check_file):
	if i == 0:
		continue
	checks.append(list(map(int, line.replace('\n', '').split(',')[1:])))

print('Output results')
output = open(output_path, 'w')
output.write('ID,Ans\n')
for i, check in enumerate(checks):
	same = clusters[check[0]] == clusters[check[1]]
	output.write(str(i) + ',' + ('1' if same else '0') + '\n')
output.close()