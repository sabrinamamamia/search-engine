#!/usr/bin/python
# -*- coding: utf-8 -*-

import ast
import json

def indexToMatrix():
	"""Converts inverted index into document-term matrix.
	Args:
		
		docList: document txt file (format: docID tf tf_idf)
	Returns:
	  doc-term matrix text
	"""
	# inverted index format: term df idf cf [posting list])
	INDEX_PATH = "data/single.txt"
	# document format: docID tf tf_idf
	DOC_PATH = "data/single-docLength.txt"
	LEXICON = "data/lexicon.txt"
	MATRIX = "data/matrix.txt"
	DIMENSIONS = "data/dimensions.txt"
	with open(INDEX_PATH) as f1, open(DOC_PATH) as f2:
		# Convert index to list and filter top 1000 terms with highest idf
		index = []
		for line in f1: 
			term = line.split("\t")[0]
			idf = float(line.split("\t")[1].split(" ")[1])
			pList = line.split("\t")[2].replace("\n", "")
			index.append([term, idf, pList])
		
		# Sort by idf ascending 
		sortedIndex = sorted(index, key=lambda x: x[1], reverse=False)
		# Filter top 1000 terms and sort alphabetically
		filteredIndex = sorted(sortedIndex[:1000], key=lambda x: x[0])

		docs = {line.split(" ")[0]: [0] * 1000 for line in f2}

		termIndex = {k[0]: v for v, k in enumerate(filteredIndex)}

		for elem in filteredIndex:
			term = elem[0]
			idf = elem[1]
			pList = ast.literal_eval(elem[2])
			for d in pList:
				docID = d[0]
				tf = int(d[1])
				termIdx = termIndex[term]
				docs[docID][termIdx] = tf * idf
		# print(docs)

	with open(MATRIX ,"w+") as f3, open(DIMENSIONS, "w+") as f4:
		f3.write(json.dumps(docs))
		f4.write(str(list(termIndex.keys())))

indexToMatrix()