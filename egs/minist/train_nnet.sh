#!/bin/bash
#begin configuration
learn_rate=0.001

bias_learn_rate=0.001

momentum=0.9

l1_penalty=0

l2_penalty=0
# data processing
minibatch_size=250

start_halving_inc=4

end_halving_inc=0.1

halving_factor=0.5

apply_norm=1

cv_number=10000

tr_number=50000

class_number=10

image_size=784

min_iters=5

max_iters=100

train_tool="smart-network-training"
# tool
#train_tool="smart-network-training"
#initial_tool="smart-network-initial"
# End configuration.
echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;
if [ $# != 6 ]; then
   echo "Usage: $0 <nnet.init> <TrainFile> <CrossvalFile> <labels-tr> <labels-cv> <LogFile>"
   echo "e.g.: $0 0.nnet train-file crossval-file labels-tr labels-cv"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi
nnet_proto=$1
TrainFile=$2
tr_labels=$3
CrossvalFile=$4
cv_labels=$5
dir=$6

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

mlp_init=$dir/nnet/nnet.init

# Skip training
[ -e $dir/final.nnet ] && echo "'$dir/nnet/final.nnet' exists, skipping training" && exit 0

##############################
# Start training
# initial the network with proto

smart-network-initial --prototype=$nnet_proto --nnet-initial=$mlp_init

# choose mlp to start with
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}

# optionally resume training from the best epoch
[ -e $dir/nnet/mlp_best ] && mlp_best=$(cat $dir/nnet/mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)
[ -e $dir/.bias_learn_rate ] && learn_rate=$(cat $dir/.bias_learn_rate)
# cross-validation on original network
$train_tool --cross-validate=1 \
 --minibatch-size=$minibatch_size  \
 --CrossValFile=$CrossvalFile \
 --cv-lable=$cv_labels \
 --cv-number=$cv_number \
 --class-number=$class_number \
 --apply-norm=$apply_norm \
 --image-size=$image_size \
 --mlp-best=$mlp_best \
 > $dir/log/iter00.initial.log || exit 1;
awk 'BEGIN{ ORS=" "; a=0; b =0}NR==FNR{if($0 ~/epoch/) {print $9; a = a + $9; b = b+1}} END{print "\n"; printf "the AvgLoss Xent is: %.2f",a/b; print "\n"}' $dir/log/iter00.initial.log >> $dir/log/iter00.initial.log
xent=$(grep "AvgLoss" $dir/log/iter00.initial.log | awk '{print $5}')
loss_type=$(grep "AvgLoss" $dir/log/iter00.initial.log | awk '{ print $3; }')
echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $xent) $loss_type"

# resume lr-halving
halving=0
[ -e $dir/.halving ] && halving=$(cat $dir/.halving)
# training
for iter in $(seq  $max_iters); do
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}

  # skip iteration if already done
  [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue 
  
  # training
  $train_tool --learn-rate=$learn_rate \
   --bias-learnrate=$bias_learn_rate \
   --l1-penalty=$l1_penalty \
   --l2-penalty=$l2_penalty \
   --apply-norm=$apply_norm \
   --minibatch-size=$minibatch_size \
   --momentum=$momentum \
   --TrainFile=$TrainFile \
   --tr-lable=$tr_labels \
   --tr-number=$tr_number \
   --class-number=$class_number \
   --image-size=$image_size \
   --mlp-best=$mlp_best \
   --mlp-next=$mlp_next \
   > $dir/log/iter${iter}.tr.log || exit 1;

   awk 'BEGIN{ ORS=" "; a=0; b =0}NR==FNR{if($0 ~/epoch/) {print $9; a = a + $9; b = b+1}} END{print "\n"; printf "the AvgLoss Xent is: %.2f",a/b; print "\n"}' $dir/log/iter${iter}.tr.log >> $dir/log/iter${iter}.tr.log
   tr_xent=$(grep "AvgLoss" $dir/log/iter${iter}.tr.log | awk '{print $5}')
   echo  "TRAIN AVG.LOSS $(printf "%.4f" $tr_xent)"
   
  # cross-validation
   $train_tool --cross-validate=1 \
    --minibatch-size=$minibatch_size  \
    --CrossValFile=$CrossvalFile \
    --cv-lable=$cv_labels \
    --cv-number=$cv_number \
    --class-number=$class_number \
    --apply-norm=$apply_norm \
    --image-size=$image_size \
    --mlp-best=$mlp_next \
   >$dir/log/iter${iter}.cv.log || exit 1;
  

awk 'BEGIN{ ORS=" "; a=0; b =0}NR==FNR{if($0 ~/epoch/) {print $9; a = a + $9; b = b+1}} END{print "\n"; printf "the AvgLoss Xent is: %.2f",a/b; print "\n"}' $dir/log/iter${iter}.cv.log >> $dir/log/iter${iter}.cv.log
xent_new=$(grep "AvgLoss" $dir/log/iter${iter}.cv.log | awk '{print $5}')
echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $xent_new) "

  # accept or reject new parameters (based on objective function)
  xent_prev=$xent

   if [ "1" == "$(awk "BEGIN{print($xent_new<$xent);}")" ]; then
    xent=$xent_new
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_xent)_cv$(printf "%.4f" $xent_new)
    mv $mlp_next $mlp_best
    echo "nnet accepted ($(basename $mlp_best))"
    #echo $mlp_best > $dir/mlp_best 
  else
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_xent)_cv$(printf "%.4f" $xent_new)_rejected
    mv $mlp_next $mlp_reject
    echo "nnet rejected ($(basename $mlp_reject))"
  fi

  # create .done file as a mark that iteration is over
  touch $dir/.done_iter$iter

    # start annealing when improvement is low
  if [ "1" == "$(awk "BEGIN{print($xent_prev < $xent+$start_halving_inc)}")" ]; then
    halving=1
    echo $halving >$dir/.halving
  fi

  # do annealing
  if [ "1" == "$halving" ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    bias_learn_rate=$(awk "BEGIN{print($bias_learn_rate*$halving_factor)}")
    echo $bias_learn_rate > $dir/.bias_learn_rate
    echo $learn_rate >$dir/.learn_rate
  fi

  # stopping criterion
  if [[ "1" == "$halving" && "1" == "$(awk "BEGIN{print($xent_prev < $xent+$end_halving_inc)}")" ]]; then
    if [[ "$min_iters" != "" ]]; then
      if [ $min_iters -gt $iter ]; then
        echo we were supposed to finish, but we continue, min_iters : $min_iters
        continue
      fi
    fi
    echo finished, too small rel. improvement $(awk "BEGIN{print($xent_prev-$xent)}")
    break
  fi



done

# select the best network
if [ $mlp_best != $mlp_init ]; then 
  mlp_final=${mlp_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nnet; )
  echo "Succeeded training the Neural Network : $dir/final.nnet"
else
  "Error training neural network..."
  exit 1
fi




