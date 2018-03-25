## COSC 488 - Information Retrieval 
## Search Engine Project

This programs pre-processes and tokenizes a TREC benchmark dataset as the first step of building an information retrieval system and search engine. After pre-processing, this engine can build a single term index, positional index of single terms, phrase index, and stem index. Then, it applies three different retrieval strategies/models and similarity measures to perform relevance ranking. The following models are used: 
* Vector Space Model using Cosine measure (tf-idf normalized for document length)
* Probabilistic model: BM25
* Language model: Query Likelihood with Dirichlet Smoothing 

To evaluate the performance, Mini-Trec created from the TREC benchmark data provided by NIST, along with Trec title queries. The qrels file has “presumably” the relevant documents to each of the queries in the query file. 

This engine has a shell interface that accepts arguments in the following format:

### Building The Index
`python3 build.py [trec-files-directory-path] [index-type] [output-dir]`

* `[trec-files-directory-path]`  is the directory containing the raw documents
* `[index-type]`  can be one of the following: `single`,  `stem`,  `phrase`, `positional`
* `[output-dir]` is the directory where index and lexicon files will be written
* Example command: `python3 build.py data/ positional output/`

### Query Processing (Report 1, Static) 
`python3 query.py [index-dir-path][query-file-path][retrieval-model][index-type][results-file]`

* `[index-dir-path]` takes the path to the directory where you store your index files (the [output] of the "build index" step).
* `[query-file-path]` path to the query file
* `[retrieval-model]` can one of the following: "cosine", "bm25", "lm"
* `[index-type]` one of the following: "single", "stem"
* `[results-file]` is the path to the results file, this file will be run with trec_eval to get the performance of your system. 
* Example: `python3 query.py ./indexes/ ./data/queryfile.txt cosine single ./results/results.txt`

### Query Processing (Report 2, Dynamic) 
`python3 query_dynamic.py [index-directory-path] [query-file-path] [results-file]`

* `[index-dir-path]` takes the path to the directory where you store your index files (the [output] of the "build index" step).
* `[query-file-path]` path to the query file
* `[results-file]` is the path to the results file, this file will be run with trec_eval to get the performance of your system. 
* Ex: `python3 query_dynamic.py ./indexes/ ./data/queryfile.txt ./results/results-dynamic.txt`

### Parser/Tokenizer Requirements
* Identifies each token – this is each single term that can be a word, a number, etc. Each token is identified as the one separated from the other token by a space, period, symbols (^,*,#,@, $…). These symbols are not stored. 
* Performs case folding and change all to lower case. 
* Identifies and store special tokens such as dates, emails, and IP addresses. 
* Identifies two and three word phrases for the phrase index
* Uses the Porter stemmer algorthim to stem the terms for the lexicon of the stem index

### Index-builder Requirements
* Create several indexes:
    - Single term index - do not include stop terms
    - Single term (including stop terms) positional index
    - Stem index
    - Phrase index
* Creates a memory constraint parameter. This parameter specifies the memory requirements in term of number of triples, i.e., amount of data can be kept in memory. Put this constraint as 1000, 10,000, and 100,000 triples. 
* Uses sort-based algorithm to create inverted index. 
* Captures system time needed to make the inverted index.


