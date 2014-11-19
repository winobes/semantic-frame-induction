# Scripts #

* `download_triarcs.sh`
    - Get the triarcs.XX-of-99.gz for XX = 00 to 98 (that totals 99 files).
    - downloaded from http://comondatastorage.googleapis.com/books/syntactic-ngrams/eng/
* `processX.sh` (there 5 so they can be run simultaniously)
    - Extract the triarc (keeping the `.gz` just in case we need it).
    - Process the data using `ProcessNGram.hs` (`output triarcs.XX-of-99.prep`).
    - Delete the extracted triarc (otherwise I run out of disk space).
* `sort.sh`
    - Create a temporary file with the names of the files we want to process (`triarcs.XX-of-99.prep`).
    - Sort the contents of all of those files alphanumerically into `all_VSOs.sorted`.
    - uses the unix command `sort` (external r-way merge).
* `ConcatVSO.hs`
    - Scans through the file and concatenates like-lines in the VSO format (summing their counts)
    - Assumes that the file *has* been sorted!


# ParseNGram.hs #

## Input ##
Syntactic n-gram files from Google.

## Output ##
Pruned VSO (verb, subject, object) files. Each VSO is separated by a newline, and
each line has the following format:

    Verb Subject Object Count

Where `Verb`, `Subject`, and `Object` may contain any non-whitespace character and
`count` may contain only digits.

Note that crucially, the output may contain lines that are duplicate `SVO`s since they
may have come from different triarcs. This holds within the processed files, but maybe
also accross the files (not sure about that part). The `sort` and `ConcatVSO` scripts
_should_ take care of that.

## Useage ##
First must be compiled using GHC (The Glasgow Haskell Compiler)

    ghc ParserNgram.hs

Then can be used as follows:

    ./ParserNgram triarcs.00-of-99

(To process multiple files, use the bash script.)

## What it does ##
For each line of the Google file, ParserNgram parses the line to a general n-gram
datatype, ignoring the counts by date. Then it checks to see if the n-gram contains
an SVO trigram. If it does, the n-gram is pruned to just the SVO trigram and written
in a simplied format (see above). 

A trigram is considered a SVO if the head word is a verb (pos-tag "VB"), and there is
a subject (dep-tag "nsubj") and an object (dep-tag "dobj") dependent on it. (Are there
other dep-tags that should qualify?) We only consider trigrams where the verb is the 
head since we may doubble count some trigrams otherwise. 
