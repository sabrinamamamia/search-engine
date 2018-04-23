## COSC 488 - Information Retrieval 
## Search Engine Project

This search engine has the following functionalities:
* Preprocesses and tokenizes a 1700-document TREC benchmark dataset
* Builds 4 types of inverted indexes and matrices (single, stem, positional, and phrase) 
* Clusters documents using an unsupervised machine learning algorithm (K-Means)* Applies 3 retrieval models (Vector Space Model, BM25, Query Likelihood with Smoothing) and performs relevance ranking 

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

### Building the Document-Term Matrix and Query Matrix 
`python3 preprocess.py [input-directory-path] [results-directory] [num-dimensions]`
* Example: python3 preprocess.py data data 1000
* `[input-data-path]` is directory with the inverted index
* `[results-directory]` is the directory where the matrix files will go
* `[num-dimensions]` is the number of dimensions the vectors will have 
NOTE: I only used single term inverted index for this part of the project. However, this system can easily be scaled out to use other inverted indexes. 

### Clustering
`python3 cluster.py [input-directory-path] [results-directory-path] [K]`
* Example: python3 cluster.py data results 10
* `[input-data-path]` is directory with the inverted index
* `[results-directory]` is the directory where the matrix files will go
* `[K]` is the number of clusters

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


