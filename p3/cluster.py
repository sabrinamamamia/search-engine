import numpy as np
import ast
import random
import math
import sys
import copy
import sys
import os
from time import time

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
	"""Calculates cosine similarity between vectors.
	Args:
		vector1: vector 1
		vector2: vector 2
	Returns:
		distance: float representing euclidean distance 
	"""
	# Euclidean distance - did not work for me
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
	while i < MAX_ITERATIONS and newSSE < finalSSE:
		finalSSE = newSSE
		if type(docs) == set:
			docs = {a:a for a in docs}

		# Assign documents to a cluster 
		for j, d in enumerate(docs): 
			# Find distance between doc and centroids
			distances = []
			for k in centroids:
				distances.append(getDistance(docTermMatrix[docs[d]], k))
			# Reassign vector to closest cluster
			oldk = getClusterID(docs[d], clusters)
			newk = distances.index(max(distances))
			if oldk != None and oldk != newk:
				clusters[oldk].remove(docs[d])
			if oldk != newk:
				clusters[newk].add(docs[d])
		SSE = 0
		# k_SSE = []
		# Recompute centroid and get SSE
		for k in range(K):
			centroids[k] = recomputeCentroid(clusters[k], docTermMatrix)
			SSE += getSSE(centroids[k], clusters[k], docTermMatrix)
			# k_SSE.append(getSSE(centroids[k], clusters[k], docTermMatrix))
		newSSE = SSE 
		i +=1
	
	# FINAL_k_SSE = []
	# for k in range(K):
		# FINAL_k_SSE.append(round(getSSE(centroids[k], clusters[k], docTermMatrix),2))
	# print("Cluster sizes:" + str([len(c) for c in clusters]))
	# print("SSE of ea cluster: " + str(FINAL_k_SSE))
	# print("SSE: " + str(round(finalSSE,2)))
	return Cluster(centroids, clusters)

def processQuery(query, centroids, docTermMatrix):
	"""Calculates the SC between each query vector and each cluster centroid
		 and picks the cluster with the highest SC 
	Args:
		query: query vector
		centroids: list of cluster centroids
	Returns:
		SCs: list of [index, distance] sorted in ascending order
	"""
	SCs = []
	for idx, centroid in enumerate(centroids):
		SCs.append([idx, getDistance(query, centroid)])
	# Highest SC = smallest distance between query and centroid
	# print(sorted(SCs, key = lambda x: x[1], reverse=True))
	return sorted(SCs, key = lambda x: x[1], reverse=True)

def generateScores(clusters, query, selected, SCs, queryIDs, docs, docTermMatrix):
	# Get SC between query and documents in selected cluster
	scores = []	
	for idx in clusters[selected]: 
		scores.append([docs[idx], getDistance(docTermMatrix[idx], query)])
	# Search other clusters if selected cluster has < 100 documents
	while len(scores) < 100 and len(SCs) > 0:
		# print("Original cluster too small, searching other clusters")
		selected = SCs.pop(0)[0]
		for idx in clusters[selected]: 
			scores.append([docs[idx], getDistance(docTermMatrix[idx], query)])
	sortedScores = sorted(scores, key = lambda x: x[1], reverse=True)
	for idx, score in enumerate(sortedScores[:100]):
		line = str(queryIDs[qidx]) + " 0 " + score[0] + " " + str(idx) + " " + str(score[1]) + " COSINE"
		resultsFile.write(line + "\n")

def analyzeCluster(clusters, centroids, docTermMatrix):
	"""Calculates size, intra-class similarity (avg distance betwegit vectors in cluster),
	 and inter-class similarity of clusters (avg distance between clusters)
	Args:
		clusters: list of clusters
	"""
	# Interclass similarity
	interClass = 0
	n = len(clusters)
	for idx, c in enumerate(clusters):
		for idx1, c1, in enumerate(clusters):
			if idx1 > idx:
				interClass += getDistance(centroids[idx], centroids[idx1])
				# print("c" + str(idx) + " <-> c" + 
				# 	str(idx1) + " " + str(round(getDistance(centroids[idx], centroids[idx1]),2)))
	print(round(interClass / ((n * (n-1)) / 2),2))

	# Intraclass similarity
	# Test only on cluster 0
	intraClass = 0
	for doc in clusters[0]:
		# print(getDistance(centroids[0], docTermMatrix[doc]))
		intraClass += getDistance(centroids[0], docTermMatrix[doc])
	print(round(intraClass / len(clusters[0]),2))

	# print(getDistance(centroids[0], docTermMatrix[doc]))

def main():
	start_time = time()
	inputPath = sys.argv[1]
	resultsPath = sys.argv[2]
	K = int(sys.argv[3])

	# Creates result output directory if necessary
	resultsPath  = resultsPath.split("/")
	resultsPath.pop(0) if resultsPath[0] == "." else resultsPath
	resultsDir = resultsPath[0] 
	if not os.path.exists(resultsDir):
		os.makedirs(resultsDir)

	if inputPath[-1] != "/":
		inputPath += "/"  
	if resultsDir[-1] != "/":
		resultsDir += "/" 

	DOCTERM_MATRIX = inputPath + "docterm-matrix.txt"
	QUERY_MATRIX = inputPath + "query-matrix.txt"
	DIMENSIONS = inputPath + "dimensions.txt"
	RESULTS = resultsDir + "results.txt"
	resultsFile = open(RESULTS, "a+")

	docTermDict = eval(open(DOCTERM_MATRIX, 'r').read())
	queryDict = eval(open(QUERY_MATRIX, 'r').read())
	docs = sorted(list(docTermDict.keys()))
	queryIDs = sorted(list(queryDict.keys()))
	docsDict = {k: v for v, k in enumerate(docs)} #k: docID, v: index
	terms = ast.literal_eval(open(DIMENSIONS).read())
	docTermMatrix = np.array([docTermDict[i] for i in docs])
	queryMatrix = np.array([queryDict[i] for i in queryIDs])

	# Cluster document collection
	clusterComponents = KMeans(K, docsDict, docTermMatrix)
	centroids = clusterComponents.centroids
	clusters = clusterComponents.clusters

	# Perform query processing on clustered collection
	for qidx, query in enumerate(queryMatrix):
		if qidx != 21:	# Query ID 374 returns 0 results, skip
			# Calculate the SC between the query vector and each cluster centroid 
			SCs = processQuery(query, centroids, docTermMatrix)
			selected = SCs.pop(0)[0]
			# Recluster if selected cluster has > 100 documents 
			if len(clusters[selected]) > 100: 
				recluster = True
				# Make size of K 75% of orginal so clusters are not too small
				reClusterComponents = KMeans(int(K * .75), clusters[selected], docTermMatrix)
				centroids1 = reClusterComponents.centroids
				clusters1 = reClusterComponents.clusters
				SCs = processQuery(query, centroids1, docTermMatrix)
				selected = SCs.pop(0)[0]

			else: 
				recluster = False
			scores = []	
			if recluster == False:
				for idx in clusters[selected]: 
					scores.append([docs[idx], getDistance(docTermMatrix[idx], query)])
			elif recluster == True:
				for idx in clusters1[selected]: 
					scores.append([docs[idx], getDistance(docTermMatrix[idx], query)])
			
			# Search other clusters if selected cluster has < 100 documents
			while len(scores) < 100 and len(SCs) > 0:
				selected = SCs.pop(0)[0]
				if recluster == False:
					for idx in clusters[selected]: 
						scores.append([docs[idx], getDistance(docTermMatrix[idx], query)])
				elif recluster == True:
					for idx in clusters1[selected]: 
						scores.append([docs[idx], getDistance(docTermMatrix[idx], query)])
			sortedScores = sorted(scores, key = lambda x: x[1], reverse=True)
			for idx, score in enumerate(sortedScores[:100]):
				line = str(queryIDs[qidx]) + " 0 " + score[0] + " " + str(idx) + " " + str(score[1]) + " COSINE"
				resultsFile.write(line + "\n")
	end_time = time()
	print('Time:   {:.3f} s'.format(end_time - start_time))

if __name__== "__main__":
	main()