rm -f "filenames.tmp"
for i in $(seq -f "%02g" 0 98)
do
  printf "triarcs.%s-of-99.prep\0" $i >> "filenames.tmp"
done
sort --output="all_VSOs.sorted" --files0-from=filenames.tmp
rm "filenames.tmp"
