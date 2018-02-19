## COSC 488 - Information Retrieval 
## Search Engine Project Design Document 

### main()
* Parse command-line arguments
* Read stop words file
* Create output directories where index and lexicon files will be written
* Read TREC data files and pass to preProcess
* Sort and merge temp files in sortAndMerge
* Build inverted index from merged list of triples

### preProcess()
* Parse each file using BeautifulSoup, a Python package for text parsing
* For each document, extract the document id and preprocess/tokenize whatever is enclosed between the <TEXT> tags
* For each document, create a dictionary termfreq that store term’s frequency. 
* For each line in document, replace escape sequences 
* To tokenize, use re.split() to split line on space, period, and symbols (^,*,#,@, $ …) for the single, positional, and stem index and used nltk’s tokenizer nltk.word_tokenize() for the phrase index, because this tokenizer keeps punctuation after parsing, which is used to determine whether a sequence of tokens was a phrase.  
* Perform case folding and change all to lower case 
* Add token to termfreq if it is not a stop word 
* After a document is processed, generate triples from termfreq in createTriples 
* Write triples to disk in writeToDisk

### Pre-processing specifics for each index
* Single - found and removed special tokens from line in findSpecialTokens()
* Positional - used a counter to assign position to every token 
* Stem - used nltk.stem.porter to perform token stemming
* Phrase - preProcessPhrase() identifies two-term and three-term phrases that do not cross stop-words, punctuation marks, and special symbols

### findSpecialTokens()
* Create regex patterns for all types of special tokens. If there is a match, perform special token-specific normalization, add token to termfreq, and then remove from line. 
* Strange email addresses are truncated so they have a valid format (e.g, BARNES.Don@EPAMAIL.EPA.GOV@IN -> BARNES.Don@EPAMAIL.EPA.GOV) 
* Abbreviations and acronyms were converted to lowercase and stripped of periods  
* Dates converted to MM-DD-YYYY format in normalizeDate()
* Date ranges are not handled
* Decimal and currency were converted to whole number and rounded up
In single term index, parts of hyphenated terms stored as well as the entire term without hyphens (e.g., hello-world-123 stored as hello, world, 123, and helloworld123) 
* File extensions and IP addresses stored with period (e.g., helloworld.pdf) 
Used list of prefixes from this website: http://www.grammar-monster.com/lessons/hyphens_in_prefixes.htm
* Original IP address regex `\b(\d{1,3}\.){3}\d{1,3}\b` incorrectly matched references to chapters/procedures (e.g., 4.3.3.1). Because these can be identified as single-digits, I changed my regex such that a digit had to repeat at least twice in order to be considered an IP address, i.e. `\b(\d{2,3}\.){3}\d{2,3}\b`

### createTriples() 
* Iterate through termfreq to create the (term, document id, term frequency) triples
* If there is a memory constraint, check length of list of triples. If the length equals memory capacity, write triples to disk and empty the list. 

### writeToDisk() 
* Writes triples to temporary files in output-dir/temp/. Number of triples in each temp file based on declared memory constraint. 

### sortAndMerge()
* Implements sort/merge-based index construction, specifically:
* Sortes temp files by term and document id 
* Perform m-way merging using priority queue of intermediate files in memory and write merged list of triples onto the disk 
* Received OSError: [Errno 24] Too many open files because my max number of open files limit was too low. To configure, ran `ulimit -n <limit>` in command line 

### buildInvertedIndex() 
* Converts list of triples to inverted index and lexicon using a dictionary 
Index files in `output-dir/indexes/`
* For the phrase index, also creates a filtered-phrase index that only contains terms with a df > 1.  
* For the positional index, the first element is the document frequency, and the rest of the elements are the positions. 
