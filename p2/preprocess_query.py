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

def findSpecialTokens(query, stops):
	"""Identify special tokens using regular expressions. If match, 
		normalize token, add to parsed query list, and remove from query. 
	Args:
	    query: query string
	    stops: list of stop words 
	Returns:
	    query with special token removed
	"""
	# Email 
	email = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
	if len(email.findall(query)) > 0:
		for token in email.findall(query):
			token = normalize(token)
			parsedQuery.append(token)
		query = email.sub('', query)

	# IP Address
	ip = re.compile(r'\b(?:\d{2,3}\.){3}\d{2,3}\b')
	if len(ip.findall(query)) > 0:
		for token in ip.findall(query):
			token = normalize(token)
			parsedQuery.append(token)
		query = ip.sub('', query)

	# Abbreviations and Acronyms
	abbrev = re.compile(r'\b[A-Z][a-zA-Z\.]{,1}[A-Z]\b\.?')
	if len(abbrev.findall(query)) > 0:
		for token in abbrev.findall(query):
			token = normalize(token)
			token = normalize(token).replace('.', '')
			parsedQuery.append(token)
		query = abbrev.sub('', query)

	# Date 
	date = re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec[.]?|January|February|March|April|May|June|July|August|September|October|November|December)\s(\d+),\s(\d+)|((\d+)(?:/|-)(\d+)(?:/|-)(\d+))')
	if len(date.findall(query)) > 0:
		for token in date.findall(query):
			token = normalizeDate(token)
			if token != None:
				parsedQuery.append(token)
		query = date.sub('', query)

	# Decimal and Currency
	decimal = re.compile(r'(§|\$)*(\d+\.\d+)')
	if len(decimal.findall(query)) > 0:
		for token in decimal.findall(query):
			token = token[0] + str(int(round(float(token[1]))))
			parsedQuery.append(token)
		query = decimal.sub('', query)

	#Hyphenated, Alpha-Digit, and Digit-Alpha terms
	hyphen = re.compile(r'(((\w+)-)+(\w+))')
	prefixes = set(['a', 'an', 'ante', 'anti', 'auto', 'circum', 'co', 'com',
	'con', 'contra', 'de', 'dis', 'en', 'ex', 'extra', 'hetero', 'homo', 
	'inter', 'intra','kilo', 'macro', 'micro', 'milli', 'non', 'pico', 'pseudo',
	'pre', 'post', 're', 'sub', 'syn', 'trans', 'tri', 'un', 'uni', 'ultra'])

	if len(hyphen.findall(query)) > 0:
		for token in hyphen.findall(query):
			token = normalize(token[0])
			split = token.split("-")
			if len(split) == 2 and split[0].isalpha() and split[1].isdigit():
				# Requirement: alphabets stored as a separate term if 3+ letters
				if len(split[0]) >= 3:
					parsedQuery.append(normalize(split[0]))
			elif len(split) == 2 and split[0].isdigit() and split[1].isalpha():
				if len(split[1]) >= 3:
					parsedQuery.append(normalize(split[1]))
			elif split[0] in prefixes:
				parsedQuery.append(normalize(split[1]))	
			else: 
				for s in split:
						if s not in stops: 
							parsedQuery.append(s)
			token = token.replace("-", "")
			parsedQuery.append(token)
		query = hyphen.sub('', query)

	# URL
	url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
	if len(url.findall(query)) > 0: 
		for token in url.findall(query):
			parsedQuery.append(token)
		query = url.sub('', query)

	# File Extension
	fileext = re.compile(r'(\w*)\.(jpg|JPG|gif|GIF|doc|DOC|pdf|PDF|html|HTML)')
	if len(fileext.findall(query)) > 0: 
		for token in fileext.findall(query):
			parsedQuery.append(token)
		query = fileext.sub('', query)
	return query

def loadStops():
	"""Load stop words"""
	with open("data/stops.txt") as f:
		stops = f.readlines()
	stops = set([x.strip() for x in stops])
	return stops

def parse(query, indexType):
	"""Reads query string and preprocesses it similar to how documents were processed
	Args: 
		query: query string 
		indexType: Type of index (single, phrase, etc)
	Returns:
		dictionary (Key: token, Value: tf)
	"""

	global parsedQuery
	parsedQuery = []
	stops = loadStops()
	query = replaceEscSeq(query)

	if (indexType == "stem"):
		stemmer = PorterStemmer()
	
	if indexType == "single": 
		query = findSpecialTokens(query, stops)

	tokens = re.split("\s|\$|\^|\*|@|\(|\)|/|○|•|\,|\?|\!|\;|\:|\`|\]|\[|&", query)
	tokens = list(filter(None, tokens))
	for token in tokens:
		token = normalize(token)
		if indexType == "single" and (token not in stops and token != ''):
			parsedQuery.append(token)
		elif indexType == "positional" and token != '':
			position += 1
			parsedQuery.append(token)	
		elif indexType == "stem" and (token not in stops and token != ''):
			token = stemmer.stem(token)
			parsedQuery.append(token)

		#Compute query term tf 
		queryDict = {}
		for term in parsedQuery:
			if term not in queryDict:
				queryDict[term] = 0
			queryDict[term] += 1
	return queryDict

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

def parsePhrase(query, stops):

	line = replaceEscSeq(query)
	tokens = nltk.word_tokenize(query)
	tokens = list(filter(None, tokens))
	tokens = [x.lower() for x in tokens]

	phrases = []
	i = j = 0
	while i < len(tokens)-1:
		phrase = ""
		while j < len(tokens) and (j-i) < 3:
			if isPhrase(stops, tokens[j]):
				phrase += tokens[j] + " "
				if (j - i) > 0:
					phrases.append(phrase[:-1])
			else:
				break
			j += 1
		i += 1
		j = i
	return phrases

# def parsePhrase(query, stops):
# 	"""Parses query to identify two/three-word phrases.
# 	Args:
# 	    query: query string 
# 	"""
# 	phrases = []
# 	query = replaceEscSeq(query)
# 	tokens = nltk.word_tokenize(query)
# 	tokens = [x.lower() for x in tokens]
# 	print(tokens)

# 	if len(tokens) == 2:
# 		if isPhrase(stops, tokens[0]) and isPhrase(stops, tokens[1]):
# 			print(tokens[0] + " " + tokens[1])
# 			phrases.append(tokens[0] + " " + tokens[1])

# 	elif len(tokens) > 2: 
# 		lastTwo = []
# 		i = j = 0
# 		phrase = ""
# 		while i < len(tokens) - 2:
# 			while (j - i) < 3 and isPhrase(stops, tokens[j]):
# 				phrase += tokens[j] + " "
# 				if j - i >= 1:
# 					phrase = phrase
# 					print(phrase[:-1])
# 					# addToDict(phrase[:-1], termfreq, None)
# 				j += 1
# 			if (j - i) < 3:
# 				i = j = j + 1
# 			else:
# 				i = j = i + 1
# 			phrase = ""
# 		if len(tokens) > 1 and isPhrase(stops, tokens[-2]) and isPhrase(stops, tokens[-1]):
# 			lastTwo.append(tokens[-2])
# 			lastTwo.append(tokens[-1])
# 		elif isPhrase(stops, tokens[-1]):
# 			lastTwo.append(tokens[-1])
# 	return phrases
