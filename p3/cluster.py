import numpy as np
import ast
import random
import math
import sys
import copy
from pyrsistent import s

def selectRandomSeeds(K, max):
	"""Pick K random points as cluster centers called centroids.
	Args:
		docs: list of docIDs 
		K: number of clusters
	Returns:
		randSeeds: set of K random docIDs
	"""
	randSeeds = set()
	while len(randSeeds) < K:
		# Select rand index between 0 and len of doc list
		seed = random.randint(0,max)	
		# Ensure seeds are unique
		if seed not in randSeeds:
			randSeeds.add(seed)
	return randSeeds

def getClusterID(docID, clusters):
	for idx, cluster in enumerate(clusters):
		if docID in cluster:
			return idx
	return None

def getDistance(doc, centroid):
	"""Calculates Euclidean distance (the square root 
		of tf*idf difference across all dimensions squared)
		between centroid and document vector.
	Args:
		doc: document vector
		centroid: centroid vector
	Returns:
		distance: float representing euclidean distance 
	"""
	distance = np.linalg.norm(centroid-doc)
	# summation = 0
	# for idx, val in enumerate(centroid):
	# 	summation += math.pow(centroid[idx] - doc[idx],2)
	return distance

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
	MAX_ITERATIONS = 20
	centroids = []														# List of centroid vectors 
	centroidsIdx = []													# List of centroid vector indicies 
	seeds = selectRandomSeeds(K, len(docs))		# K random indicies 
	for s in seeds:
		centroids.append(docTermMatrix[s])
		centroidsIdx.append(s)

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
		# print("Itr: " + str(i))
		finalSSE = newSSE
		finalClusters = clusters 		#Shallow copy
		finalCentroids = centroids

		# Assign documents to a cluster 
		for d in docs: 
			# Find distance between doc and centroids
			distances = []
			for k in centroids:
				distances.append(getDistance(docTermMatrix[docs[d]], k))
			
			# Reassign vector to closest cluster
			oldk = getClusterID(docs[d], clusters)
			newk = distances.index(min(distances))
			if oldk != None and oldk != newk:
				clusters[oldk].remove(docs[d])
			if oldk != newk:
				clusters[newk].add(docs[d])

		SSE = 0
		k_SSE = []
		# Recompute centroid and get SSE
		for k in range(K):
			centroids[k] = recomputeCentroid(clusters[k], docTermMatrix)
			SSE += getSSE(centroids[k], clusters[k], docTermMatrix)
			k_SSE.append(getSSE(centroids[k], clusters[k], docTermMatrix))
		# print("SSE of ea cluster: " + str(k_SSE))
		# print("SSE: " + str(SSE))
		newSSE = SSE 
		i +=1

	FINAL_k_SSE = []
	for k in range(K):
		FINAL_k_SSE.append(getSSE(centroids[k], clusters[k], docTermMatrix))
	print("SSE of ea cluster: " + str(FINAL_k_SSE))
	print("Final SSE: " + str(finalSSE))

def main():
	MATRIX = "data/matrix.txt"
	DIMENSIONS = "data/dimensions.txt"
	K = 15

	dataDict = eval(open(MATRIX, 'r').read())
	docs = sorted(list(dataDict.keys()))
	docsDict = {k: v for v, k in enumerate(docs)} #k: docID, v: index
	terms = ast.literal_eval(open(DIMENSIONS).read())
	docTermMatrix = np.array([dataDict[i] for i in docs])

	KMeans(K, docsDict, docTermMatrix)


if __name__== "__main__":
	main()