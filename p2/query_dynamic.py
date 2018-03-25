#!/usr/bin/python
# -*- coding: utf-8 -*-

import preprocess 
import query as query_static
import os
import sys
import collections
import ast
import math
from time import time

def processQuery(query, indexType, index):
	"""Processes and sends query to the BM25 retrieval model. 
	Logical flow is similar to that in query.py main() function.  
	Args: 
		query: query string 
		indexType: type of index (single, stem, phrase, positional)
		index: dictionary (key: term, value: Term object)
	Returns:
		docLength dictionary (key: docID, value: DocLength object)
	"""
	scores = {}
	docLength = query_static.getDocLength(indexType)

	N = len(docLength)
	C = query_static.getC(docLength)
	avgDocLength = C / N 

	if indexType == "phrase":
		query = [query]
	elif indexType == "positional":
		query = query.split(" ")
	elif indexType == "single":
		query = preprocess.parse(query, "single")

	for term in query:
		if term in index: 
			q_tf = 1
			pList = index[term].pList 
			t_idf = float(index[term].idf)

			for doc in pList: 
				docid = doc[0]
				if indexType == "positional":
					doc_tf = len(ast.literal_eval(pList[0][1]))
				else: 
					doc_tf = int(doc[1].split(" ")[0])

				score = query_static.BM25(n=index[term].df, doc_tf=doc_tf, q_tf=q_tf, 
				N=N, doclen=docLength[docid].tf, avgdoclen=avgDocLength)
				if docid in scores:
						scores[docid] += score
				else:
					scores[docid] = score
	return scores

def getMax(ptrs, allPos):
	"""Finds position with largest value
	Args: 
		ptrs = list of pointer indices   
		allPos: list of positional lists
	Returns:
	  position with largest value
	"""
	maxPos = 0
	for i, posList in enumerate(allPos):
		maxPos = max(maxPos, posList[ptrs[i]])
	return maxPos

def longestList(allPos):
	"""Finds positional list with longest length
	Args: 
		allPos: list of positional lists
	Returns:
	  Index of element in allPos
	"""
	maxLength = max(len(l) for l in allPos)
	for l in allPos:
		if len(l) == maxLength:
			return allPos.index(l)

def isPhrase(allPos):
	"""Determines if query contains valid phrase based on whether  
	terms appear within 30 positions of each other
	Args: 
		allPos: list of positional lists
	Returns:
	  True if phrase found, else False  
	"""
	# Assign pointers to first element 
	ptrs = []
	for p in allPos:
		ptr = 1
		ptrs.append(ptr)

	longest = longestList(allPos)
	while ptrs[longest] < len(allPos[longest]):
		maxPos = getMax(ptrs, allPos)
		# All terms must appear within 30 pos of each other
		lower = maxPos - 15
		upper = maxPos + 15

		# Move pointers to position greater than the lower bound (skip pointer)
		for i, posList in enumerate(allPos):
			while ptrs[i] < len(posList) and posList[ptrs[i]] < lower:
				ptrs[i] += 1
			# ptr reach end of list, no phrase found
			if ptrs[i] == len(allPos[i]):
				return False

		matchCount = 0
		for i, posList in enumerate(allPos):
			if posList[ptrs[i]] in range(lower, upper):
				matchCount += 1
		# If all positions are within the range, then a phrase is found
		if matchCount == len(allPos):
			return True
	return False

def intersect(pLists):
	"""Finds intersection of posting lists based on document id 
	Args: 
		pLists: list of posting lists in format '[[docID, [positional list]], ...]
	Returns:
	 dictionary (key: document id, value: list of positional lists)
	"""
	# Convert posting lists to dicts (key: doc id, val: position list)
	plDicts = []
	plSets = []
	for pl in pLists:
		plDict = {k[0]: ast.literal_eval(k[1]) for k in pl}
		plDicts.append(plDict)
		pl_set = set(plDict.keys())
		plSets.append(pl_set)
	# Get documents that exist in all posting lists
	resultSet = plSets[0]
	for s in plSets[1:]:
		resultSet = s.intersection(resultSet)
	final = {}
	for doc in resultSet:
		temp = []
		for plDict in plDicts:
			temp.append(plDict[doc])
		final[doc] = temp
	return final

def main():
	# [index-directory-path] [query-file-path] [results-file]
	# python3 query_dynamic.py ./indexes/ ./data/queryfile.txt ./results/results-dynamic.txt
	# ./trec_eval qrel.txt results/results.txt

	start_time = time()

	indexPath = sys.argv[1]
	queryPath = sys.argv[2]
	resultsPath = sys.argv[3] 

	if indexPath[-1] != "/":
		indexPath += "/"  
	if resultsPath[-1] != "/":
		resultsPath += "/" 

	# Creates result output directory if necessary
	resultsPath  = resultsPath.split("/")
	resultsPath.pop(0) if resultsPath[0] == "." else resultsPath
	resultsDir = resultsPath[0] 
	resultsFile = resultsPath[1]
	if not os.path.exists(resultsDir):
		os.makedirs(resultsDir)

	# Load inverted index and document length dictionary into memory 
	resultsFile = open(resultsDir + "/" + resultsFile, "w+")
	phraseIndex = query_static.getIndex(indexPath, "phrase-filtered")

	# Preprocess queries 
	queryNums = []
	queries = []
	with open(queryPath) as f:
		for line in f.readlines():
			if line.startswith("<num>"):
				num = line.replace("<num> Number: ", "").replace("\n", "").rstrip()
				queryNums.append(num)
			elif line.startswith("<title>"):
				num = line.replace("<title> Topic: ", "").replace("\n", "").rstrip()
				queries.append(num)

		# Finds phrases in query
		for qIdx, q in enumerate(queries):
			scores = {}
			stops = preprocess.loadStops()
			phrases = preprocess.parsePhrase(q, stops)

			phraseCount = 0
			for phrase in phrases:
				# Checks if phrase is in the filtered phrase index (phrases with df > 1)
				if phrase in phraseIndex:
					# print("sending to phrase")
					scores = processQuery(phrase, "phrase", phraseIndex)
				# Send query to positional index
				else:
					posIndex = query_static.getIndex(indexPath, "positional")
					terms = phrase.split(" ")
					if all(term in posIndex for term in terms):
						pLists = []
						for term in terms: 
							pLists.append(posIndex[term].pList)
						potentialDocs = intersect(pLists)
						for d in potentialDocs:
							if isPhrase(potentialDocs[d]) == True:
								phraseCount += 1 
						if phraseCount >= 1:
							# print("sending to positional")
							positionalIndex = query_static.getIndex(indexPath, "positional")
							scores = processQuery(phrase, "positional", positionalIndex)
							continue

			# If not enough documents found then use single term index
			if len(scores) < 100 or phraseCount == 0: 
				singleIndex = query_static.getIndex(indexPath, "single")
				scores1 = processQuery(q, "single", singleIndex)
				scoresUnion = {**scores, **scores1}
				scoresIntersection = set(scores).intersection(set(scores1))
				for doc in scoresIntersection:
					# Get weighted average of phrase and single idx scores
					scoresUnion[doc] = (scores[doc] + scores1[doc]) / 2
				scores = scoresUnion

			topScores = collections.Counter(scores).most_common(100)
			for i, s in enumerate(topScores):
				docID = str(s[0])
				rank = str(i)
				score = str(s[1])
				resultsFile.write(queryNums[qIdx] + " 0 " + docID + " " + rank + " " + 
					score + " BM25 " + "\n")

	end_time = time()
	print('Query Processing:   {:.3f} s'.format(end_time - start_time))

if __name__== "__main__":
	main()