# process_all.sh #

1. Download:
    - Get the verbargs.XX-of-99.gz for XX = 00 to 98 (total 99 files).
    - downloaded from http://comondatastorage.googleapis.com/books/syntactic-ngrams/eng/
    
2. Filter the n-gram files:
    - Extract the verbarg (keeping the `.gz` file just in case we need it).
    - Process the data using `FilterNGrams.hs` (output `verbargs.XX-of-99.prep`).
    - Delete the extracted verbarg.
    
3. Put everything in one sorted file:
    - Create a temporary file with the names of the files we want to process (`verbargs.XX-of-99.prep`).
    - Sort the contents of all of those files alphanumerically into `all_VSOs.sorted`.
    - uses the unix command `sort` (external r-way merge).
    
4. Concatenate like VSOs
    - Concatenate `all_VSOs.sorted` using `ConcatVSO.hs`
    - Assumes that the file has already been sorted!


# FilterNGrams.hs #

## Input ##
Syntactic n-gram files from Google.

## Output ##
Pruned VSO (verb, subject, object) files. Each VSO is separated by a newline, and
each line has the following format:

    Verb Subject Object Count

Where `Verb`, `Subject`, and `Object` may contain any non-whitespace character and
`count` may contain only digits.

Note that crucially, the output may contain lines that are duplicate `SVO`s since they
may have come from different verbargs. This holds within the processed files, but maybe
also accross the files (not sure about that part). The `sort` and `ConcatVSO` scripts
_should_ take care of that.

## Useage ##
First must be compiled using GHC (The Glasgow Haskell Compiler)

    ghc FilterNGrams.hs

Then can be used as follows:

    ./FilterNGrams verbargs.00-of-99

(To process multiple files, use the bash script.)

## What it does ##
For each line of the Google file, it parses the line to a general n-gram
datatype, ignoring the counts by date. Then it checks to see if the n-gram contains
an SVO trigram. If it does, the n-gram is pruned to just the SVO trigram and written
in a simplied format (see above). 

A trigram is considered a SVO if the head word is a verb (pos-tag "VB"), and there is
a subject (dep-tag "nsubj") and an object (dep-tag "dobj") dependent on it. (Are there
other dep-tags that should qualify?) We only consider trigrams where the verb is the 
head since we may doubble count some trigrams otherwise. 

# ConcatVSO.hs #

## Input ##
A VSO file (as outputted by FilterNGrams.hs).

## Output ##
A concatenated VSO file (the output file will have the same name as the input
with a `.concat` extension).

For exmaple the lines

    buy john bread 18
    buy john bread 30
    eat mary rice 28

become

    buy john bread 48
    eat mary rice 28

Note that we assume that the file has already been sorted alpha-numerically. 

## Useage ##
Compile using GHC (as above). Pass the file to be concatenated:

    ./ConcatVSO all_VSOs.sorted

## What it does ##
Goes through the file line by line and concatenates the current line with the previous
one (by adding their counts) if the verb, subject and object agree. When the next line
doesn't agree with the previous one (or there are no more lines), it writes the current 
line to the output file and keeps going. Note this has the side-effect of reversing the 
order of the sort. 
