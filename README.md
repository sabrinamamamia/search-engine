## COSC 488 - Information Retrieval 
## Search Engine Project
### Part 1: Pre-Processing Documents & Building Inverted Index

This programs pro-processes and tokenizes a TREC benchmark dataset as the first step of building an an efficient and scalable information retrieval system and search engine. This engine includes a single term index, positional index of single terms, phrase index, and stem index. 

The program has a shell interface that accepts arguments in the following format:

`python3 build.py [trec-files-directory-path] [index-type] [output-dir]`

* Example command: `python3 build.py data/ positional output/`
* `[trec-files-directory-path]`  is the directory containing the raw documents
* `[index-type]`  can be one of the following: `single`,  `stem`,  `phrase`, `positional`
* `[output-dir]` is the directory where index and lexicon files will be written

### Parser/Tokenizer Requirements
* Identifies each token – this is each single term that can be a word, a number, etc. Each token is identified as the one separated from the other token by a space, period, symbols (^,*,#,@, $…). These symbols are not stored. 
* Performs case folding and change all to lower case. 
* Identifies and store special tokens such as dates, emails, and IP addresses. 
* Identifies two and three word phrases for the phrase index
* Uses the Porter stemmer algorthim to stem the terms for the lexicon of the stem index

### Index-builder Requirements:
* Create several indexes:
    - Single term index - do not include stop terms
    - Single term (including stop terms) positional index
    - Stem index
    - Phrase index
* Creates a memory constraint parameter. This parameter specifies the memory requirements in term of number of triples, i.e., amount of data can be kept in memory. Put this constraint as 1000, 10,000, and 100,000 triples. 
* Uses sort-based algorithm to create inverted index. 
* Captures system time needed to make the inverted index.

