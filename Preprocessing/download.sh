for i in $(seq -f "%02g" 0 98)
do
  wget http://commondatastorage.googleapis.com/books/syntactic-ngrams/eng/triarcs.$i-of-99.gz
done
