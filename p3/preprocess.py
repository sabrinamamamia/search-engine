#!/usr/bin/python
# -*- coding: utf-8 -*-

import ast
import json
import preprocess_query 
import sys
import os
from time import time

def indexToMatrix(inputPath, resultsPath, numDimensions):
	"""Converts single term inverted index into document-term matrix and query matrix. 
	Args:
		inputPath: data input directory path
		resultsPath: directory where result files will go
	Returns:
	  doc-term text file
	  query matrix text file
	  dimensions text file
	"""
	# inverted index format: term df idf cf [posting list])
	INDEX = inputPath + "single.txt"
	# document format: docID tf tf_idf
	DOC = inputPath + "single-docLength.txt"
	LEXICON = inputPath + "lexicon.txt"
	QUERY = inputPath + "queries.txt"
	QUERY_MATRIX = resultsPath + "query-matrix.txt"
	DOCTERM_MATRIX = resultsPath + "docterm-matrix.txt"
	DIMENSIONS = resultsPath + "dimensions.txt"

	with open(INDEX) as indexFile, open(DOC) as docFile, open(QUERY) as qFile:
		# Convert index to list and filter top n terms with highest idf
		index = []
		indexDict = {}
		for line in indexFile: 
			term = line.split("\t")[0]
			idf = float(line.split("\t")[1].split(" ")[1])
			pList = line.split("\t")[2].replace("\n", "")
			index.append([term, idf, pList])
			indexDict[term] = [idf, pList]

		# Preprocess queries 
		queryNums = []
		queries = []
		for line in qFile.readlines():
			if line.startswith("<num>"):
				num = line.replace("<num> Number: ", "").replace("\n", "").rstrip()
				queryNums.append(num)
			elif line.startswith("<title>"):
				num = line.replace("<title> Topic: ", "").replace("\n", "").rstrip()
				queries.append(num)
		
		queryTerms = set()
		for qIdx, q in enumerate(queries):
			query = preprocess_query.parse(q, "single")	# (term: tf) dict 
			queryID = queryNums[qIdx]
			for term in query:
				queryTerms.add(term)
		
		# Sort by idf ascending 
		sortedIndex = sorted(index, key=lambda x: x[1], reverse=False)

		# Filter out idf > 1 
		# sortedIndex1 = sorted(sortedIndex, key=lambda x: x[1] >= 1, reverse=False)
		sortedIndex1 = list(filter(lambda x: x[1] >= 1, sortedIndex))

		# Filter top n terms and sort alphabetically
		filteredIndex = sorted(sortedIndex1[:numDimensions])
		filteredIndexSet = {s[0] for s in filteredIndex}

		for term in queryTerms:
			if term not in filteredIndexSet and term in indexDict:
				filteredIndex.append([term, indexDict[term][0], indexDict[term][1]])

		# Create empty doc-term dict to be the doc-term matrix
		docTermMatrix = {line.split(" ")[0]: [0] * len(filteredIndex) for line in docFile}
		# Create (term: index) dict so weights placed in correct index in matrix
		termIndex = {k[0]: v for v, k in enumerate(filteredIndex)}
		indexDict = {}	# (term: idf) dict
		# Place tfidf weights in doc-term matrix
		for elem in filteredIndex:
			term = elem[0]
			idf = elem[1]
			pList = ast.literal_eval(elem[2])
			indexDict[term] = idf
			for d in pList:
				docID = d[0]
				tf = int(d[1])
				termIdx = termIndex[term]
				docTermMatrix[docID][termIdx] = tf * idf

		queryMatrix = {num: [0] * len(filteredIndex) for num in queryNums}
		for qIdx, q in enumerate(queries):
			query = preprocess_query.parse(q, "single")	# (term: tf) dict 
			queryID = queryNums[qIdx]
			for term in query:
				if term in termIndex:
					termIdx = termIndex[term]
					tf = query[term]
					idf = indexDict[term]
					queryMatrix[queryID][termIdx] = tf * idf

	with open(DOCTERM_MATRIX ,"w+") as f3, open(QUERY_MATRIX, "w+") as f4, open(DIMENSIONS, "w+") as f5:
		f3.write(json.dumps(docTermMatrix))
		f4.write(json.dumps(queryMatrix))
		f5.write(str(list(termIndex.keys())))


def main():

	# python3 preprocess.py [input-directory-path] [results-directory-path] [num-dimensions]
	inputPath = sys.argv[1]
	resultsPath = sys.argv[2]
	numDimensions = int(sys.argv[3])

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

	indexToMatrix(inputPath, resultsDir, numDimensions)

if __name__== "__main__":
	main()

