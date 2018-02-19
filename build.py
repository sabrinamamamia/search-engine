#!/usr/bin/python
# -*- coding: utf-8 -*-

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

def normalizeDate(match):
	"""Change date tokens to MM-DD-YYYY format.
	Args:
	    match: tokens that match date regex in findSpecialTokens.
	Returns:
	    formated date string  
	"""
	months = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, 
	"Aug": 8, "Sep": 9,  "Oct": 10, "Nov": 11, "Dec": 12, "January": 1,  "February": 2, 
	"March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, 
	"September": 9, "October": 10, "November": 11, "December": 12}
	month30Day = set([4, 6, 9, 11])
	month31Day = set([1, 3, 5, 7, 8, 10, 12])

	# Date format: Month Name DD, YYY 
	if match[0] != '':
		month = months[match[0].replace(".", "")]
		day = int(match[1])
		year = int(match[2])

	# Date format: MM/DD/YYYY || MM-DD-YYYY
	elif match[3] != '':
		month = int(match[4])
		day = int(match[5])
		year = int(match[6])

	# Rule out invalid dates
	if (month < 1 or month > 12) or (month == 2 and (day < 1 or day > 29)) or \
		(month in month30Day and (day < 1 or day > 30)) or \
		(month in month31Day and (day < 1 or day > 31)):
			return

	# Change YY to YYYY
	if len(str(year)) == 2:
		if year < 19:
			year = int('20' + str(year))
		else:
			year = int('19' + str(year))
	if year < 1918 or year > 2018:
		return

	month = str(month) if len(str(month)) == 2 else '0' + str(month)
	day = str(day) if len(str(day)) == 2 else '0' + str(day)
	return (month + '-' + day + '-' + str(year))

def normalize(token):
	"""Perform case folding and punctuation stripping.
	Args:
	    token: single token
	Returns:
	    normalized token 
	"""
	token = token.lower()							# Lowercase words
	token = token.replace(',', '')		# Remove commas 
	token = token.strip(punctuation)	# Remove punctuation
	return token

def replaceEscSeq(line):
	"""Process escape sequences.
	Args:
	    line: line from data file data.
	Returns:
	    line with "&<seq>; replaced with symbol it represents 
	"""
	match = re.search(r'(&[a-z]+;)', line)
	if match:
		line = line.replace('&blank;', '&').replace('blank;', '&') \
		.replace('&cir;','○').replace('&hyph;', '-').replace('&sect;','§') \
		.replace('&times;', '×').replace('&racute;', 'r')
	return line

def addToDict(token, termfreq, position):
	"""Add term to a temporary dict. Key: term, Value: frequency (+ positions)
	Args:
	    token: single token 
	    termfreq: dict
	    position: int if indexType == positional, else None 
	Returns:
	    line with "&<seq>; replaced with symbol it represents 
	"""
	if position == None:
		if token not in termfreq: 
			termfreq[token] = 0
		termfreq[token] += 1

	# For positional index
	else:		
		if token not in termfreq: 
			termfreq[token] = []
			termfreq[token].append(0)
		termfreq[token][0] += 1
		termfreq[token].append(position)

def findSpecialTokens(line, stops, termfreq):
	"""Identify special tokens using regular expressions. If match, 
		normalize token, add to dictionary, and remove from line. 
	Args:
	    line: line in data file 
	    termfreq: dict
	Returns:
	    line with special token removed
	"""
	# Email 
	email = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
	if len(email.findall(line)) > 0:
		for token in email.findall(line):
			token = normalize(token)
			addToDict(token, termfreq, None)
		line = email.sub('', line)

	# IP Address
	ip = re.compile(r'\b(?:\d{2,3}\.){3}\d{2,3}\b')
	if len(ip.findall(line)) > 0:
		for token in ip.findall(line):
			token = normalize(token)
			addToDict(token, termfreq, None)
		line = ip.sub('', line)

	# Abbreviations and Acronyms
	abbrev = re.compile(r'\b[A-Z][a-zA-Z\.]{,1}[A-Z]\b\.?')
	if len(abbrev.findall(line)) > 0:
		for token in abbrev.findall(line):
			token = normalize(token)
			token = normalize(token).replace('.', '')
			addToDict(token, termfreq, None)
		line = abbrev.sub('', line)

	# Date 
	date = re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec[.]?|January|February|March|April|May|June|July|August|September|October|November|December)\s(\d+),\s(\d+)|((\d+)(?:/|-)(\d+)(?:/|-)(\d+))')
	if len(date.findall(line)) > 0:
		for token in date.findall(line):
			token = normalizeDate(token)
			if token != None:
				addToDict(token, termfreq, None)
		line = date.sub('', line)

	# Decimal and Currency
	decimal = re.compile(r'(§|\$)*(\d+\.\d+)')
	if len(decimal.findall(line)) > 0:
		for token in decimal.findall(line):
			token = token[0] + str(int(round(float(token[1]))))
			addToDict(token, termfreq, None)
		line = decimal.sub('', line)

	#Hyphenated, Alpha-Digit, and Digit-Alpha terms
	hyphen = re.compile(r'(((\w+)-)+(\w+))')
	prefixes = set(['a', 'an', 'ante', 'anti', 'auto', 'circum', 'co', 'com',
	'con', 'contra', 'de', 'dis', 'en', 'ex', 'extra', 'hetero', 'homo', 
	'inter', 'intra','kilo', 'macro', 'micro', 'milli', 'non', 'pico', 'pseudo',
	'pre', 'post', 're', 'sub', 'syn', 'trans', 'tri', 'un', 'uni', 'ultra'])

	if len(hyphen.findall(line)) > 0:
		for token in hyphen.findall(line):
			token = normalize(token[0])
			split = token.split("-")
			if len(split) == 2 and split[0].isalpha() and split[1].isdigit():
				# Requirement: alphabets stored as a separate term if 3+ letters
				if len(split[0]) >= 3:
					addToDict(normalize(split[0]), termfreq, None)
			elif len(split) == 2 and split[0].isdigit() and split[1].isalpha():
				if len(split[1]) >= 3:
					addToDict(split[1], termfreq, None)
			elif split[0] in prefixes:
				addToDict(split[1], termfreq, None)		
			else: 
				for s in split:
						if s not in stops: 
							addToDict(s, termfreq, None)
			token = token.replace("-", "")
			addToDict(token, termfreq, None)
		line = hyphen.sub('', line)

	# URL
	url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
	if len(url.findall(line)) > 0: 
		for token in url.findall(line):
			addToDict(token, termfreq, None)
		line = url.sub('', line)

	# File Extension
	fileext = re.compile(r'(\w*)\.(jpg|JPG|gif|GIF|doc|DOC|pdf|PDF|html|HTML)')
	if len(fileext.findall(line)) > 0: 
		for token in fileext.findall(line):
			addToDict(token, termfreq, None)
		line = fileext.sub('', line)
	return line

def createTriples(termfreq, docid, triples, memory, outputDir):
	"""Create (term, document id, term frequency) triples
	Args:
	    termfreq: dictionary 
	    docid: document id
	    triples: list of triples
	Returns:
	    triples
	"""
	# Unlimited Memory 
	if memory == "unlimited":
		for key,value in termfreq.items():
			triples.append(str(key)+"\t"+str(docid)+"\t"+str(value))
		return triples

	# Memory Constraint
	for key,value in termfreq.items():
		if len(triples) == memory: 
			writeToDisk(triples, memory, outputDir)
			triples = []
		triples.append(str(key)+"\t"+str(docid)+"\t"+str(value))
	return triples


def writeToDisk(triples, memory, outputDir):
	"""Writes triples to temporary txt files in output-dir/temp/.
	Args:
	    triples: list of triples
	"""
	i = 0
	tempFileLineCount = 0

	while os.path.exists(outputDir + "/temp/temp%s" % i):
		i += 1

	if os.path.exists(outputDir + "/temp/temp%s" % str(i - 1)):
		with open(outputDir + "/temp/temp%s" % str(i - 1), "r") as f:
		  for j, l in enumerate(f):
		      pass
		  tempFileLineCount = j + 1
		  if tempFileLineCount < memory:
		  	tempFile = open(outputDir + "/temp/temp%s" % str(i - 1), "a") 
		  else:
		  	tempFile = open(outputDir + "/temp/temp%s" % i, "w")	
	else:
		tempFile = open(outputDir + "/temp/temp%s" % i, "w")	

	for t in triples:
		if tempFileLineCount < memory: 
			tempFile.write(str(t)+"\n")
		else:
			tempFile = open(outputDir + "/temp/temp%s" % i, "w")
			tempFileLineCount = 0
			tempFile.write(str(t)+"\n")
		tempFileLineCount += 1
	tempFile.close()

def preProcess(data, triples, stops, indexType, memory, outputDir):
	"""Parses files to identify tokens and their frequency.
	Args:
	    data: TREC file
	"""
	if memory != "unlimited":
		triples = [] 

	soup = BeautifulSoup(data, "lxml")
	doc = soup.find_all('doc')

	if (indexType == "stem"):
		stemmer = PorterStemmer()

	for d in doc:
		termfreq = {}
		position = 0 
		docID = d.find('docno').text.strip()
		text = d.find('text').text

		lines = list(filter(None, text.splitlines()))
		for i, line in enumerate(lines):
			line = replaceEscSeq(line)
			if indexType == "single": 
				line = findSpecialTokens(line, stops, termfreq)  
			# tokens = nltk.word_tokenize(line)
			tokens = re.split("\s|\$|\^|\*|@|\(|\)|/|○|•|\,|\?|\!|\;|\:|\`|\]|\[|&", line)
			tokens = list(filter(None, tokens))
			for token in tokens:
				token = normalize(token)
				if indexType == "single" and (token not in stops and token != ''):
					addToDict(token, termfreq, None)
				elif indexType == "positional" and token != '':
					position += 1
					addToDict(token, termfreq, position)	
				elif indexType == "stem" and (token not in stops and token != ''):
					token = stemmer.stem(token)
					addToDict(token, termfreq, None)
		triples = createTriples(termfreq, docID, triples, memory, outputDir)
	if memory != "unlimited":
		writeToDisk(triples, memory, outputDir)
	return triples

def isPhrase(stops, token):
	"""Checks that token is not a symbol or stop word.
	Args:
	    data: TREC file
	"""
	symbol = set([",","(", ")", ".", "?", "!", "'", ";", "...", " ", "", ":", "",
		"-", "@","$","^","*","@","/", "○", "•", "``", "''", "&", "[", "]", "%", "#"])
	if token not in stops and token not in symbol:
		return True
	return False

def preProcessPhrase(data, triples, stops, indexType, memory, outputDir):
	"""Parses files to identify two/three-word phrases.
	Args:
	    data: TREC file
	"""
	if memory != "unlimited":
		triples = []
	soup = BeautifulSoup(data, "lxml")
	doc = soup.find_all('doc')

	for d in doc:
		termfreq = {}
		position = 0 
		docID = d.find('docno').text.strip()
		text = d.find('text').text

		lastTwo = []
		lines = list(filter(None, text.splitlines()))
		for i, line in enumerate(lines):
			line = replaceEscSeq(line)
			tokens = nltk.word_tokenize(line)
			tokens = list(filter(None, tokens))
			tokens = [x.lower() for x in tokens]

			# Handle edge case: phrase on two lines
			if isPhrase(stops, tokens[0]):
				if len(lastTwo) == 2:
					phrase = " ".join(lastTwo) + " " + tokens[0]
					phrase = phrase
					addToDict(phrase, termfreq, None)
				if len(lastTwo) == 1:
					phrase = lastTwo[0] + " " + tokens[0]
					phrase = phrase
					addToDict(phrase, termfreq, None)
			if len(tokens) > 1 and isPhrase(stops, tokens[0]) and isPhrase(stops, tokens[1]):
				if len(lastTwo) == 2:
					phrase = lastTwo[1] + " " + tokens[0] + " " + tokens[1]
					phrase = phrase
					addToDict(phrase, termfreq, None)
				if len(lastTwo) == 1:
					phrase = lastTwo[0] + " " + tokens[0] + " " + tokens[1]
					phrase = phrase
					addToDict(phrase, termfreq, None)
			lastTwo = []

			i = j = 0
			phrase = ""
			while i < len(tokens) - 2:
				while (j - i) < 3 and isPhrase(stops, tokens[j]):
					phrase += tokens[j] + " "
					if j - i >= 1:
						phrase = phrase
						addToDict(phrase[:-1], termfreq, None)
					j += 1
				if (j - i) < 3:
					i = j = j + 1
				else:
					i = j = i + 1
				phrase = ""
			if len(tokens) > 1 and isPhrase(stops, tokens[-2]) and isPhrase(stops, tokens[-1]):
				lastTwo.append(tokens[-2])
				lastTwo.append(tokens[-1])
			elif isPhrase(stops, tokens[-1]):
				lastTwo.append(tokens[-1])
		triples = createTriples(termfreq, docID, triples, memory, outputDir)
	if memory != "unlimited":
		writeToDisk(triples, memory, outputDir)
	return triples
	#364039 # of triples

def sortAndMerge(outputDir):
	"""Implements sort/merge-based index construction
	"""
	if os.path.exists(outputDir + "temp/.DS_Store"):
		os.remove(outputDir + "temp/.DS_Store")

	# Sort temp files by term and document id 
	for f in os.listdir(outputDir + "/temp/"):
		lines = open(outputDir + "/temp/" + f, "r")
		sorted_file = sorted(lines, key=lambda line: (line.split()[0], line.split()[1]))
		with open(outputDir + "/temp/" + f, "w") as f:
			f.writelines(sorted_file)
	
	#M-way merge of intermediate files
	heap = [] 
	out = open(outputDir + "/merged-triples.txt", "w")

	# Received OSError: [Errno 24] Too many open files because 
	# my max number of open files limit was too low. To configure, 
	# run "ulimit -n <limit>"
	files = [open(outputDir + "/temp/" + f) for f in os.listdir(outputDir + "/temp")]

	# Use priority queue to find min triple among all files 
	while len(files) > 0:
		for f in files: 
			line = f.readline()
			line = line.replace('\n', '')
			if line == "":
				files.remove(f)
			else:
				heapq.heappush(heap, line)
		out.write(heapq.heappop(heap) + "\n")
	while heap:
		out.write(heapq.heappop(heap) + "\n")

def buildInvertedIndex(indexType, triples, memory, outputDir):
	"""Converts list of triples to inverted index
	"""
	index = {}
	indexFile = open(outputDir + "/indexes/" + indexType + ".txt", "w")
	lexiconFile = open(outputDir + "/lexicon.txt", "w")

	if memory == "unlimited":
		for t in triples:
			t = t.split("\t")
			if t[0] not in index:
				index[t[0]] = []
			index[t[0]].append([t[1], t[2].replace("\n", "")])
		for key, value in sorted(index.items()):
			indexFile.write(str(key) + " -> " + str(value) + "\n")
			lexiconFile.write(str(key) + "\n")
		indexFile.close()
		lexiconFile.close()

	else: 
		triples = open(outputDir + "/merged-triples.txt", "r")
		for t in triples.readlines():
			t = t.split("\t")
			if t[0] not in index:
				index[t[0]] = []
			index[t[0]].append([t[1], t[2].replace("\n", "")])

			if len(index) == memory:
				# Write all but last term in dictionary to file in case 
				# additional triples are part of last term's posting list
				for key, value in sorted(index.items())[:-1]:
					indexFile.write(str(key) + " -> " + str(value) + "\n")
					lexiconFile.write(str(key) + "\n")
					del index[key]
		for key, value in sorted(index.items()):
			indexFile.write(str(key) + " -> " + str(value) + "\n")
			lexiconFile.write(str(key) + "\n")
		indexFile.close()
		lexiconFile.close()

	if indexType == "phrase":
		filteredFile = open(outputDir + "/indexes/" + indexType + "-filtered.txt", "w")
		unfilteredFile = open(outputDir + "/indexes/" + indexType + ".txt", "r")
		for line in unfilteredFile.readlines():
			posting_list = ast.literal_eval(line.split(" -> ")[1])
			if len(posting_list) > 1:
				filteredFile.write(line)
		filteredFile.close()
		unfilteredFile.close()

def main():
	#build [trec-files-directory-path] [index-type] [output-dir]
	global memory
	global stops
	global outputDir
	global indexType

	# Parse command-line arguments
	trecFileDirPath = sys.argv[1]  
	indexType = sys.argv[2] 
	outputDir = sys.argv[3] 
	if trecFileDirPath[-1] != "/":
		trecFileDirPath += "/" 
	if outputDir[-1] != "/":
		outputDir += "/"

	memory = "unlimited"
	triples = []

	# Load stop words 
	with open("stops.txt") as f:
		stops = f.readlines()
	stops = set([x.strip() for x in stops])

	# Create output directories
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)
		os.makedirs(os.path.join(outputDir, "temp"))
		os.makedirs(os.path.join(outputDir, "indexes"))

	# Remove old temp files, if any
	for f in os.listdir(outputDir + "/temp"):
		os.remove(outputDir + "/temp/" + f)

	for f in os.listdir(trecFileDirPath):
		data = open(trecFileDirPath + f, "r")
		if indexType == "phrase":
			triples = preProcessPhrase(data, triples, stops, indexType, memory, outputDir)
		else:
			triples = preProcess(data, triples, stops, indexType, memory, outputDir)
	if memory != "unlimited":
		sortAndMerge(outputDir)
	buildInvertedIndex(indexType, triples, memory, outputDir)

def timer():
	SETUP_CODE = '''
from __main__ import preProcess
from __main__ import preProcessPhrase
from __main__ import sortAndMerge
from __main__ import buildInvertedIndex
import os
import re
import sys
import collections
import heapq
import ast
import nltk
from bs4 import BeautifulSoup
from string import punctuation'''
	
	TEST_CODE = '''
global memory
global stops
global outputDir
global indexType

# Parse command-line arguments
trecFileDirPath = sys.argv[1]  
indexType = sys.argv[2] 
outputDir = sys.argv[3] 

memory = 1000
triples = []

# Load stop words 
with open("stops.txt") as f:
	stops = f.readlines()
stops = set([x.strip() for x in stops])

# Create output directories
if not os.path.exists(outputDir):
	os.makedirs(outputDir)
	os.makedirs(os.path.join(outputDir, "temp"))
	os.makedirs(os.path.join(outputDir, "indexes"))

# Remove old temp files, if any
for f in os.listdir(outputDir + "/temp"):
	os.remove(outputDir + "/temp/" + f)

for f in os.listdir(trecFileDirPath):
	data = open(trecFileDirPath + f, "r")
	if indexType == "phrase":
		triples = preProcessPhrase(data, triples, stops, indexType, memory, outputDir)
	else:
		triples = preProcess(data, triples, stops, indexType, memory, outputDir)
# if memory != "unlimited":
# 	sortAndMerge(outputDir)
# buildInvertedIndex(indexType, triples, memory, outputDir)'''

	TEST_CODE_MERGE = '''
global memory
global stops
global outputDir
global indexType

# Parse command-line arguments
trecFileDirPath = sys.argv[1]  
indexType = sys.argv[2] 
outputDir = sys.argv[3]

sortAndMerge(outputDir)
	'''

	times = timeit.repeat(setup = SETUP_CODE,
												stmt = TEST_CODE,
												repeat = 1,
												number = 1)

	print('Time: {}'.format(min(times)))  

if __name__== "__main__":
	main()
	# timer()
	
	

