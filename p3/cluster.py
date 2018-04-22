import numpy as np
import ast
import random
import math
import sys
import copy
from sklearn.metrics.pairwise import cosine_similarity


class Cluster:
	def __init__(self, centroids, clusters):
		self.centroids = centroids # List of centroids (vectors)
		self.clusters = clusters # List of clusters (sets of vector indexes)

def selectRandomSeeds(K, max):
	"""Pick K random points as cluster centers called centroids.
	Args: 
		K: number of clusters
		max: max random number to select
	Returns:
		randSeeds: set of K random docIDs
	"""
	randSeeds = set()
	while len(randSeeds) < K:
		# Select rand index between 0 and len of doc list
		seed = random.randint(0,max-1)	
		# Ensure seeds are unique
		if seed not in randSeeds:
			randSeeds.add(seed)
	return randSeeds

def getClusterID(docID, clusters):
	for idx, cluster in enumerate(clusters):
		if docID in cluster:
			return idx
	return None

def getDistance(vector1, vector2):
	"""Calculates Euclidean distance (the square root 
		of tf*idf difference across all dimensions squared)
		between two vectors.
	Args:
		vector1: vector 1
		vector2: vector 2
	Returns:
		distance: float representing euclidean distance 
	"""
	# distance = np.linalg.norm(vector1-vector2)
	# # summation = 0
	# # for idx, val in enumerate(centroid):
	# # 	summation += math.pow(centroid[idx] - doc[idx],2)
	# return distance

	# Cosine similarity version
	cos_sim = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
	return cos_sim

def getSSE(centroid, cluster, docTermMatrix):
	"""Calculates the sum of the squared error of cluster k.
	Args:
		centroid: centroid vector
		cluster: set of doc vector indicies
	Returns:
		sum of square errors of cluster k
	"""	
	SSE = 0
	for doc in cluster:
		SSE += getDistance(docTermMatrix[doc], centroid)
	return SSE

def recomputeCentroid(cluster, docTermMatrix):
	"""Recomputes centroid as the average squared distance of all documents in cluster.
	Args:
		cluster: set of document indexes
		docTermMatrix: document-term matrix
	Returns:
		centroid: vector with average squared distance across all dimensions 
	"""
	C = len(cluster)
	vectors = []
	for idx in cluster:
		vectors.append(docTermMatrix[idx])
	summation = np.sum(vectors, axis=0)
	centroid = np.divide(summation, C)
	return centroid

def KMeans(K, docs, docTermMatrix):
	"""Implementation of K-Means clustering algorithm 
	Args:
		K: number of clusters
		doc: (docID, vector index) dictionary
		docTermMatrix: document-term matrix
	Returns:
		Cluster: object containing final centroids and clusters  
	"""
	MAX_ITERATIONS = 20
	centroids = []														# List of centroid vectors 
	centroidsIdx = []													# List of centroid vector indicies 
	seeds = selectRandomSeeds(K, len(docs))		# K random indicies 

	for s in seeds:
		centroids.append(docTermMatrix[s])
		centroidsIdx.append(s)
	# print(seeds)

	# Initialize clusters (list of doc vector index sets)
	clusters = [{c} for c in centroidsIdx]		

	# Terminating conditions: 
	# 1) fixed number of iterations 
	# 2) no more reduction in sum of squared error (SSE)
	i = 0
	finalSSE = 100000
	newSSE = finalSSE - 1
	# TODO: Why does SSE increase after i + 1 iterations
	while i < MAX_ITERATIONS and newSSE < finalSSE:
	# while i < MAX_ITERATIONS:
		# print("Itr: " + str(i))
		finalSSE = newSSE
		finalClusters = clusters 		#Shallow copy
		finalCentroids = centroids
		if type(docs) == set:
			docs = {a:a for a in docs}

		# Assign documents to a cluster 
		for j, d in enumerate(docs): 
			# Find distance between doc and centroids
			distances = []
			for k in centroids:
				distances.append(getDistance(docTermMatrix[docs[d]], k))
			# print(distances)
			# Reassign vector to closest cluster
			oldk = getClusterID(docs[d], clusters)
			newk = distances.index(min(distances))
			if oldk != None and oldk != newk:
				clusters[oldk].remove(docs[d])
				# print("moving doc to new clustre")
			if oldk != newk:
				clusters[newk].add(docs[d])
			# print([len(c) for c in clusters])
		SSE = 0
		k_SSE = []
		# Recompute centroid and get SSE
		for k in range(K):
			centroids[k] = recomputeCentroid(clusters[k], docTermMatrix)
			# print("change between centroids")
			# print(getDistance(old, centroids[k]))
			SSE += getSSE(centroids[k], clusters[k], docTermMatrix)
			k_SSE.append(getSSE(centroids[k], clusters[k], docTermMatrix))
		# print("SSE of ea cluster: " + str(k_SSE))
		# print("SSE: " + str(SSE))
		newSSE = SSE 
		i +=1
	
	FINAL_k_SSE = []
	for k in range(K):
		FINAL_k_SSE.append(round(getSSE(centroids[k], clusters[k], docTermMatrix),2))
	print("Cluster sizes:" + str([len(c) for c in clusters]))
	print("SSE of ea cluster: " + str(FINAL_k_SSE))
	print("Final SSE: " + str(round(finalSSE,2)))
	return Cluster(centroids, clusters)

def processQuery(query, centroids, docTermMatrix):
	"""Calculates the SC between each query vector and each cluster centroid
		 and picks the cluster with the highest SC 
	Args:
		query: query vector
		centroids: list of cluster centroids
	Returns:
		maxSCs: list of indices of centroid with highest SC for each query
	"""
	SCs = []
	for centroid in centroids:
		SCs.append(getDistance(query, centroid))
	# Highest SC = smallest distance between query and centroid
	print(SCs)
	return SCs.index(min(SCs))

def analyzeCluster(clusters, centroids, docTermMatrix):
	"""Calculates size, intra-class similarity (avg distance between vectors in cluster),
	 and inter-class similarity of clusters (avg distance between clusters)
	Args:
		clusters: list of clusters
	"""
	interClass = 0
	intraClass = []	
	sizes = []
	n = len(clusters)

	for idx1, c1 in enumerate(clusters): 
		# To determine interclass, compare every clusters centroids with every other
		# cluster then divide by n(n - 1) / 2
		for idx2, c2 in enumerate(clusters):
			interClass += getDistance(centroids[idx1], centroids[idx2])
		# Intraclass: compare every document with centroid and divide by total # of doc ??
		intra = 0
		for doc in cluster:
			intra += getDistance(centroids[idx], docTermMatrix[doc])
		intraClass.append(intra/len(cluster))
	interclass = interclass / (n(n-1) / 2)

def main():
	DOCTERM_MATRIX = "data/docterm-matrix.txt"
	QUERY_MATRIX = "data/query-matrix.txt"
	DIMENSIONS = "data/dimensions.txt"
	RESULTS = "data/test-results.txt"
	resultsFile = open(RESULTS, "a+")
	K = 8

	docTermDict = eval(open(DOCTERM_MATRIX, 'r').read())
	queryDict = eval(open(QUERY_MATRIX, 'r').read())
	docs = sorted(list(docTermDict.keys()))
	queryIDs = sorted(list(queryDict.keys()))
	docsDict = {k: v for v, k in enumerate(docs)} #k: docID, v: index
	terms = ast.literal_eval(open(DIMENSIONS).read())
	docTermMatrix = np.array([docTermDict[i] for i in docs])
	queryMatrix = np.array([queryDict[i] for i in queryIDs])

	# for qidx, query in enumerate(queryMatrix):
	# 	scores = [] 
	# 	for i, doc in enumerate(docTermMatrix):
	# 		scores.append([docs[i], getDistance(query, doc)])
	# 	sortedScores = sorted(scores, key = lambda x: x[1], reverse=True)
	# 	for idx, score in enumerate(sortedScores[:100]):
	# 		line = str(queryIDs[qidx]) + " 0 " + score[0] + " " + str(idx) + " " + str(score[1]) + " EUCLID"
	# 		resultsFile.write(line + "\n")

	# Q = np.array([0,0,0,0,0,.176,0,0,.477,0,.176])
	# D = np.array([0,0,.477,0,.477,.176,0,0,0,.176,0])

	# print(terms.index("domestic"))
	# print(queryMatrix[0])
	# print(docTermMatrix[docsDict["FR940617-2-00223"]])
	# print(getDistance(queryMatrix[0],docTermMatrix[docsDict["FR940617-2-00223"]]))

	# print(len(terms))

	# for idx, score in enumerate(sortedScores):
	# 	line = str(queryIDs[qidx]) + " 0 " + score[0] + " " + str(idx) + " " + str(score[1]) + " EUCLID"
	# 	resultsFile.write(line + "\n")

	# Cluster document collection
	clusterComponents = KMeans(K, docsDict, docTermMatrix)
	centroids = clusterComponents.centroids
	clusters = clusterComponents.clusters

	for c in clusters:
		if docsDict["FR940617-2-00223"] in c:
			print ("in cluster " + str(clusters.index(c)))

	# highestSC = processQuery(queryMatrix[0], centroids, docTermMatrix)
	# KMeans(K, clusters[highestSC], docTermMatrix)
	# print(queryMatrix[0])
	highestSC = processQuery(queryMatrix[0], centroids, docTermMatrix)
	print("highest SC cluster" + str(highestSC))

	# for docid in clusters[highestSC]:
		# print(docs[docid])
	# print(docsDict["FR940617-2-00223"])
	print(clusters[highestSC])

	scores = []
	for idx in clusters[highestSC]: 
		print(getDistance(docTermMatrix[idx], queryMatrix[0]))
		scores.append([docs[idx], getDistance(docTermMatrix[idx], queryMatrix[0])])
	# print(scores)
	sortedScores = sorted(scores, key = lambda x: x[1], reverse=True)
	for idx, score in enumerate(sortedScores[:100]):
		line = str("265 0 " + score[0] + " " + str(idx) + " " + str(score[1]) + " COSINE")
		resultsFile.write(line + "\n")

	# Perform query processing on clustered collection
	# Calculate the SC between the query vector and each cluster centroid 
	# for qidx, query in enumerate(queryMatrix):
	# 	highestSC = processQuery(query, centroids, docTermMatrix)
		# print(len(clusters[highestSC]))
		# while len(clusters[highestSC]) > 100:
		# 	clusterComponents = KMeans(K - 3, clusters[highestSC], docTermMatrix)
		# 	centroids = clusterComponents.centroids
		# 	clusters = clusterComponents.clusters
		# 	highestSC = processQuery(query, centroids, docTermMatrix)
		# scores = []
		# for idx in clusters[highestSC]: 
		# 	scores.append([docs[idx], getDistance(docTermMatrix[idx], query)])
		# sortedScores = sorted(scores, key = lambda x: x[1], reverse=True)
		# # print(str([s for s in sortedScores]))
		# for idx, score in enumerate(sortedScores):
		# 	line = str(queryIDs[qidx]) + " 0 " + score[0] + " " + str(idx) + " " + str(score[1]) + "COSINE"
		# 	resultsFile.write(line + "\n")

if __name__== "__main__":
	main()