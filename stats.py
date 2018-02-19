#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import sys
import collections
import heapq
import ast
from bs4 import BeautifulSoup
from string import punctuation
import nltk

def computeStats():
	indexType = "phrase"
	outputDir = "output"
	numTerms = 0
	maxDF = 0
	minDF = 1
	meanDF = 0
	medianDF = 0
	medianDFCount = 0
	maxDFterm = ""
	minDFterm = ""
	sumDF = 0
	dfCount = {}

	indexFile = open(outputDir + "/indexes/" + indexType + ".txt", "r")
	for line in indexFile.readlines():
		numTerms += 1
		posting_list = ast.literal_eval(line.split(" -> ")[1])
		df = len(posting_list)
		if df > maxDF:
			maxDF = df
			maxDFterm = line.split(" -> ")[0]
		if df <= minDF:
			minDF = df
			minDFterm = line.split(": ")[0]
		if df not in dfCount:
			dfCount[df] = 0
		dfCount[df] += 1
		if dfCount[df] > medianDFCount:
			medianDF = df
			medianDFCount = dfCount[df]
		sumDF += df
	meanDF = sumDF/numTerms

	print("Lexicon size: " + str(numTerms))
	print("Max DF: " + str(maxDF))
	print("Max DF Term: " + str(maxDFterm))
	print("Min DF: " + str(minDF))
	print("Min DF Term: " + str(minDFterm))
	print("Mean DF: " + str(meanDF))
	print("Median DF: " + str(medianDF))

computeStats()






