
DIR="verbargs"
function file {
  printf "$DIR/verbargs.%02g-of-99" $1
}

# Make the directory for verbargs (if it doesn't exsit)
if [ ! -d $DIR ]; then mkdir $DIR; fi

# Compile FilterNGram if it doesn't already exist.
if [ ! -f "FilterNGrams" ]; then ghc FilterNGrams; fi

# Compile ConcatVSO if it doesn't already exist.
if [ ! -f "ConcatVSO" ]; then ghc FilterNGrams; fi

function process {
  # Download the .gz if we don't have it already.
  if [ ! -f "$1.gz" ];
  then
    "wget http://ommondatastorage.googleapis.com/books/syntactic-ngrams/eng/$1.gz"
  fi
  # Unzip and filter for VSO n-grams
  if [ ! -f "$1.gz" ];
  then
    gunzip -f --keep "$1.prep" 
    ./FilterNGrams $1 
  fi
  rm -f $1 
}

# Do four of  at a time to speed things up.
for i in $(seq  0 24); do process $(file $i); done &
for i in $(seq 25 49); do process $(file $i); done &
for i in $(seq 50 74); do process $(file $i); done &
for i in $(seq 75 98); do process $(file $i); done
wait

# Sort all the resulting VSO files.
rm -f "filenames.tmp"
for i in $(seq 0 98); do printf "$(file $i).prep\0" >> "filenames.tmp"; done
echo "Sorting everything into all_VSOs.sorted..."
sort --output="all_VSOs.sorted" --files0-from=filenames.tmp
rm "filenames.tmp"

# Concatenate like VSOs
echo "Concatenating like VSOs in all_VSOs.sorted..."
echo "$(date -u +"%F %T %Z") Found $(wc -l all_VSOs.sorted) VSOs in the n-grams files" >> log
./ConcatVSO all_VSOs.sorted
echo "$(date -u +"%F %T %Z") Found $(wc -l all_VSOs.sorted.concat) unique VSOs." >> log
