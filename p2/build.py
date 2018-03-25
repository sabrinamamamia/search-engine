#!/usr/bin/python
# -*- coding: utf-8 -*-

import preprocess 
import timeit
import os
import re
import sys
import collections
import heapq
import ast
import nltk
from bs4 import BeautifulSoup
from string import punctuation
from nltk.stem.porter import *

def main():
	global stops
	global outputDir
	global indexType
	global N 
	N = 1768

	with open("queryfile.txt") as f:
		queries = [e for e in f.readlines() if e.startswith('<title>')]
		queries = [q.replace("<title> Topic: ", "").replace("\n", "").replace("\t", "").rstrip() for q in queries]
		print(queries)
		for q in queries:
			score = {} 
			score = 

	# print("hello")
	preprocess.test()

if __name__== "__main__":
	main()