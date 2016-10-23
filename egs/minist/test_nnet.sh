#!/bin/bash

#Begin congiguration.
# data processing

minibatch_size=250
test_number=10000
apply_norm=0
image_size=784
# tool

#End configuration.
echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0  <TestFile>  <labels-te>  <LogFile>"
   echo " e.g.: $0  test-file labels-te logdir"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

test_file=$1
test_label=$2
dir=$3

# check final.nnet exit!
#[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' doesn't exists, testing fail" && exit 1

mlp_best=$dir/final.nnet
##############################
echo "================start testing===================="
smart-network-testing --minibatch-size=$minibatch_size  \
		   --TestFile=$test_file \
		   --test-label=$test_label \
		   --apply-norm=$apply_norm \
		   --image-size=$image_size \
		   --test-number=$test_number \
 		   --mlp-best=$mlp_best \
 		   2> $dir/log/testing.log || exit 1;

echo "================finished testing================="

