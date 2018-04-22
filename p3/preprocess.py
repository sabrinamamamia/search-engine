#!/usr/bin/python
# -*- coding: utf-8 -*-

import ast
import json
import preprocess_query 

def indexToMatrix():
	"""Converts inverted index into document-term matrix and query matrix. 
	Returns:
	  doc-term text file
	  query matrix text file
	"""
	# inverted index format: term df idf cf [posting list])
	INDEX = "data/single.txt"
	# INDEX = "data/stem.txt"
	# document format: docID tf tf_idf
	DOC = "data/single-docLength.txt"
	# DOC = "data/stem-docLength.txt"
	LEXICON = "data/lexicon.txt"
	QUERY = "data/queries.txt"
	QUERY_MATRIX = "data/query-matrix.txt"
	DOCTERM_MATRIX = "data/docterm-matrix.txt"
	# DOCTERM_MATRIX = "data/stem-docterm-matrix.txt"
	DIMENSIONS = "data/dimensions.txt"
	with open(INDEX) as indexFile, open(DOC) as docFile, open(QUERY) as qFile:
		
		# # Convert index to list and filter top 5000 terms with highest idf
		# index = []
		# for line in indexFile: 
		# 	term = line.split("\t")[0]
		# 	idf = float(line.split("\t")[1].split(" ")[1])
		# 	pList = line.split("\t")[2].replace("\n", "")
		# 	index.append([term, idf, pList])
		
		# # Sort by idf ascending 
		# sortedIndex = sorted(index, key=lambda x: x[1], reverse=False)

		# # Filter out idf > 1 
		# # sortedIndex1 = sorted(sortedIndex, key=lambda x: x[1] >= 1, reverse=False)
		# sortedIndex1 = list(filter(lambda x: x[1] >= 1, sortedIndex))

		# # Filter top 5000 terms and sort alphabetically
		# filteredIndex = sorted(sortedIndex1[:5000], key=lambda x: x[0])

		# # Create empty doc-term dict to be the doc-term matrix
		# docTermMatrix = {line.split(" ")[0]: [0] * 5000 for line in docFile}
		# # Create (term: index) dict so weights placed in correct index in matrix
		# termIndex = {k[0]: v for v, k in enumerate(filteredIndex)}
		# indexDict = {}	# (term: idf) dict
		# # Place tfidf weights in doc-term matrix
		# for elem in filteredIndex:
		# 	term = elem[0]
		# 	idf = elem[1]
		# 	pList = ast.literal_eval(elem[2])
		# 	indexDict[term] = idf
		# 	for d in pList:
		# 		docID = d[0]
		# 		tf = int(d[1])
		# 		termIdx = termIndex[term]
		# 		docTermMatrix[docID][termIdx] = tf * idf

		# # Preprocess queries 
		# queryNums = []
		# queries = []
		# for line in qFile.readlines():
		# 	if line.startswith("<num>"):
		# 		num = line.replace("<num> Number: ", "").replace("\n", "").rstrip()
		# 		queryNums.append(num)
		# 	elif line.startswith("<title>"):
		# 		num = line.replace("<title> Topic: ", "").replace("\n", "").rstrip()
		# 		queries.append(num)
		# queryMatrix = {num: [0] * 5000 for num in queryNums}
		# queryTerms = {}
		# for qIdx, q in enumerate(queries):
		# 	query = preprocess_query.parse(q, "single")	# (term: tf) dict 
		# 	queryID = queryNums[qIdx]
		# 	for term in query:
		# 		queryTerms.add(term)
		# 		if term in termIndex:
		# 			termIdx = termIndex[term]
		# 			tf = query[term]
		# 			idf = indexDict[term]
		# 			queryMatrix[queryID][termIdx] = tf * idf
		# print(queryMatrix)

		# Convert index to list and filter top 5000 terms with highest idf
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

		# Filter top 5000 terms and sort alphabetically
		filteredIndex = sorted(sortedIndex1[:5000])
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

indexToMatrix()