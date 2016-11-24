#!/bin/bash

echo "
Experiment Seting:
	1: Network type : DNN
	2: Training criterion : Cross-Entropy
	3: Network configure :
"
apply_norm=0

batch_size=250

class_number=10

learn_rate=0.001

momentum=0.95

bias_learn_rate=0.001

l2_penalty=0.00001

is_train=1

dir=/home/tao/Works/Water

TrainFile=$dir/Data/minist/train_set

TestFile=$dir/Data/minist/test_set

CrossValiFile=$dir/Data/minist/dev_set

nnet_proto_file=$dir/egs/minist/exp/nnet.proto.sigmoid

TrainLabel=$dir/Data/minist/train_label

CrossValLabel=$dir/Data/minist/dev_label

TestLabel=$dir/Data/minist/test_label

logdir=$dir/egs/minist/exp

echo ================================================
echo "     The DNN Cross-Entropy Training           "
echo ================================================

./train_nnet.sh --minibatch-size $batch_size --learn-rate $learn_rate \
	--bias-learn-rate $bias_learn_rate \
	--apply-norm $apply_norm \
	--momentum $momentum \
	--is-train $is_train \
	--class-number $class_number \
	--l2-penalty $l2_penalty \
	$nnet_proto_file $TrainFile $TrainLabel $CrossValiFile $CrossValLabel $logdir || exit 1;


echo ================================================
echo "    The DNN Testing (based on Cross-Entropy)  "
echo ================================================

./test_nnet.sh --minibatch-size $batch_size --apply-norm $apply_norm \
	$TestFile $TestLabel $logdir|| exit 1;
