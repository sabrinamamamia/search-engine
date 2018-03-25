#!/usr/bin/python
# -*- coding: utf-8 -*-

import preprocess_query as preprocess 
import os
import sys
import collections
import ast
import math
from time import time

class Term:
	def __init__(self, df, idf, cf, pList):
		self.df = df # Document freq
		self.idf = idf # Inverse document freq
		self.cf = cf # Collection freq
		self.pList = pList #Posting list

class DocLength: 
	def __init__(self, tf, tf_idf):
		self.tf = tf # Sum of all tf 
		self.tf_idf = tf_idf # Sum of all (tf*idf)^2

def getIndex(indexPath, indexType):
	"""Reads txt file containing relevant index and coverts it to a dictionary
	Args: 
		indexType: Type of index (single, phrase, stem, positional)
	Returns:
		index dictionary (key: term, value: Term object)
	"""
	index = {}
	with open(indexPath + indexType + ".txt") as f: 
		for line in f:
			name = line.split("\t")[0]
			df = int(line.split("\t")[1].split(" ")[0])
			idf = float(line.split("\t")[1].split(" ")[1])
			cf = int(line.split("\t")[1].split(" ")[2])
			pList = ast.literal_eval(line.split("\t")[2].replace("\n", ""))
			index[name] = Term(df, idf, cf, pList)
	return index

def getDocLength(indexType):
	"""Reads txt file containing document lengths and coverts it to a dictionary
	Args: 
		indexType: Type of index (single, phrase, stem, positional)
	Returns:
		docLength dictionary (key: docID, value: DocLength object)
	"""
	docLength = {} 
	if indexType == "phrase": 
		filtered = "filtered"
	else:
		filtered = ""
	with open("data/" + indexType + filtered + "-docLength.txt") as f:
		for line in f:
			docID = line.split(" ")[0]
			tf = line.split(" ")[1]
			tf_idf = line.split(" ")[2]
			docLength[docID] = DocLength(int(tf), float(tf_idf))
		return docLength

def getC(docLength):
	"""Gets total number of terms in collection
	Args: 
		docLength: dictionary (key: docID, value: DocLength object) 
	Returns:
		C: count of total tf 
	"""
	C = 0
	for d in docLength:
		C += docLength[d].tf
	return C

def BM25(n, doc_tf, q_tf, N, doclen, avgdoclen):
	"""Implementation of probabilistic model BM25. Computes similarity
	scores between query and a document. 
	Args: 
		n: number of documents with term
		doc_tf: document term frequency
		q_tf: query term frequency
		N: number of documents 
		doclen: document length
		avgdoclen: average length of document in collection
	Returns:
		similarity score between query and document i SC(Q, Di). 
	"""
	# k1, k2, and b are parameters to be empirically determined
	k1 = 1.2
	k2 = 700
	b = 0.75
	K = k1 * ((1 - b) + b * (float(doclen)/float(avgdoclen))) 
	w = math.log((N - n + 0.5) / (n + 0.5)) #idf 
	normalizedD = (((k1 + 1) * doc_tf) / (K + doc_tf))
	normalizedQ= (((k2 + 1) * q_tf) / (k2 + q_tf))
	return w * normalizedD * normalizedQ

def LM(doc_tf, u, cf, C, D):
	"""Implementation of Query Likelihood with Dirichlet Smoothing. 
	Computes similarity scores between query and a document. 
	Args: 
		doc_tf: document term frequency
		u: smoothing factor 
		cf: collection frequency of term 
		C: total number of terms in colelction
		D: total number of documents in collection
	Returns:
		similarity score between query and document i SC(Q, Di). 
	"""
	numerator = float(doc_tf) + float(u) * (float(cf) / float(C))
	denominator = float(D) + float(u)
	return numerator/denominator

def main():
	# [index-directory-path] [query-file-path] [retrieval-model] [index-type] [results-file]
	# python3 query.py indexes/ data/queryfile.txt cosine stem results/results.txt
	# python3 query.py indexes/ data/queryfile.txt lm single results/results.txt
	# ./trec_eval qrel.txt results/results.txt

	start_time = time()

	indexPath = sys.argv[1]
	queryPath = sys.argv[2]
	retrievalModel = sys.argv[3].lower()
	indexType = sys.argv[4]
	resultsPath = sys.argv[5] 

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
	index = getIndex(indexPath, indexType)
	docLength = getDocLength(indexType)

	N = len(docLength)
	C = getC(docLength)
	avgDocLength = C / N 

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

		# Send queries to retrieval model
		for qIdx, q in enumerate(queries):
			scores = {}
			query = preprocess.parse(q, indexType)
			queryLen = 0
			for term in query:
				q_tf = query[term]
				if term in index:
					pList = index[term].pList 
					t_idf = float(index[term].idf)
					# LM: docsWithTerm distinguishes docs with tf > 0 
					docsWithTf = set()
					for doc in pList: 
						docid = doc[0]
						doc_tf = int(doc[1].split(" ")[0])
						docsWithTf.add(docid)

						if retrievalModel == "cosine":
							score = (doc_tf * t_idf) * (q_tf * t_idf)

						elif retrievalModel == "bm25":
							score = BM25(n=index[term].df, doc_tf=doc_tf, q_tf=q_tf, 
								N=N, doclen=docLength[docid].tf, avgdoclen=avgDocLength)
						elif retrievalModel == "lm":
							score = LM(doc_tf=float(doc_tf), u=avgDocLength, cf=index[term].cf, C=C, D=docLength[docid].tf)
						
						# Cosine and BM25 use summation, LM uses product 
						if docid in scores:
							if retrievalModel == "cosine" or retrievalModel == "bm25":
								scores[docid] += score
							elif retrievalModel == "lm":
								scores[docid] *= score
						else:
							scores[docid] = score

					# LM: score documents that don't contain term
					if retrievalModel == "lm":
						tmp = set(docLength.keys())
						docsWithoutTf = set(tmp).difference(docsWithTf)
						score = LM(doc_tf=0, u=avgDocLength, cf=index[term].cf, C=C, D=docLength[docid].tf)
						for docid in docsWithoutTf:
							if docid in scores:
								scores[docid] *= score 
							else:
								scores[docid] = score

				# Cosine: calculate query length
				queryLen += pow(q_tf * t_idf,2)

			# Cosine: normalize for document length 
			if retrievalModel == "cosine":
				for doc in scores: 
					scores[doc] = round(scores[doc] / math.sqrt(docLength[doc].tf_idf * queryLen), 8)

			topScores = collections.Counter(scores).most_common(100)
			# print(topScores)
			for i, s in enumerate(topScores):
				docID = str(s[0])
				rank = str(i)
				score = str(s[1])
				resultsFile.write(queryNums[qIdx] + " 0 " + docID + " " + rank + " " + 
					score + " " + str(retrievalModel.upper()) + "\n")
		end_time = time()
		print('Query Processing:   {:.3f} s'.format(end_time - start_time))

if __name__== "__main__":
	main()