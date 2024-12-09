#!/usr/bin/env bash
#
# Create a csv file an output file from any of the codes.
# to use:
#
# (path to this file)/extract2csv.sh outputfile > outputfile.csv
#

# Check the number of arguments
if [ "$#" -ne 1 ]; then
   echo
   echo "Usage: $0 INPUTFILE > Outfile.csv." >&2
   echo
   exit 1
fi

# Check the output file exists
if ! [ -e "$1" ]; then
     echo
     echo "The \"$1\" not found." >&2
     echo
     exit 2
fi

# A breakdown of what each stage in the pipeline is attempting to do.
# if the file format changes some of these lines will have to be 
# tweaked.
#
# tail -n +N skip the first N lines
# grep -v exclude lines that have two consequitive dashes
# grep -A N "Block" search for block and include following N lines
# paste -d "," - - - join 3 lines (dashes) with a "," delimiter
# tr -s " " replaces multiple spaces with one, 
# sed 's/A/B/' substitute text A with B - this is basically removing text 
# and replacing it with a comma.

# Print the column labels
echo "BlockNumber","Gsums","GsumsLength","zsum","alphvalue","CPUTime"

tail -n +11 "$1" | \
         grep -v '\-\-' | \
         grep -v '^#' | \
         grep -A 3 "Block" | \
         paste -d "," - - - | \
         tr -s " "| \
         grep "^Block" | \
         sed 's/Block number= //' | \
         sed 's/ Number of Gsums=/,/' | \
         sed 's/ Length of Gsums=/,/' | \
         sed 's/ Type of Gsums=/,/' | \
         sed 's/,zsum after this block=/,/' | \
         sed 's/ alpha value at end of this block=/,/' | \
         sed 's/,alpha value at end of this block=/,/' | \
         sed 's/ Cpu time needed for this block=/,/'

