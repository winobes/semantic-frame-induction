# this can be split into separate scripts to run concurently
for i in $(seq -f "%02g" 00 98) # by edititng this line
do
  gunzip triarcs.$i-of-99.gz --keep
  ./ParseNGram process triarcs.$i-of-99
  rm triarcs.$i-of-99
done
