#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17 exp_test"
    exit
fi

# run command
data=$1
exp_path=$2

# import data specific settings: slang, tlang, num_gpus
source $exp_path/scripts/config-$data.sh

# 0. Prerequisite
from_seg_path=$exp_path/$data-sent.segmented.$slang-$tlang
from_bin_path=$exp_path/$data-sent.binarized.$slang-$tlang

# check DA experiment
da_exp_path=$exp_path/subexp_da
da_run_path=$da_exp_path/run-sent
if [ ! -d $da_run_path ]; then
  echo `date`, The DA experiment is a prerequisite for augmentation experiment: $da_run_path
  exit -1
fi

# start MT experiment
mt_exp_path=$exp_path/subexp_mt
mkdir -p $mt_exp_path

# 1. Prepare MT data
# DONE-FLAG: skip if it has been done, otherwise clear the folders
done_file=$mt_exp_path/done.$data-sent.prepare-mt-data
if [ -f $done_file ]; then
  echo `date`, Skip prepare-mt-data since it has been done on $(<$done_file).
else
  echo `date`, Prepare MT data for sent-level model...
  da_seg_path=$da_exp_path/$data-sent.segmented.$slang-$tlang
  da_bin_path=$da_exp_path/$data-sent.binarized.$slang-$tlang
  da_res_path=$da_run_path/$data-sent.results.$slang-$tlang

  mt_seg_path=$mt_exp_path/$data-sent.segmented.$slang-$tlang
  mt_bin_path=$mt_exp_path/$data-sent.binarized.$slang-$tlang
  mkdir -p $mt_seg_path $mt_bin_path

  rm $mt_seg_path/* -rf
  rm $mt_bin_path/* -rf

  cp $from_seg_path/* $mt_seg_path/. -rf
  for ((part=1; part<=$num_gpus; part++)); do
    trainpart=train-part$part
    subset=test$part
    cat $da_seg_path/$trainpart.$tlang.origion >> $mt_seg_path/train.$tlang
    cat $da_res_path/$subset.seg.gen >> $mt_seg_path/train.$slang
  done

  python -m fairseq_cli.preprocess --task translation_doc --source-lang $slang --target-lang $tlang \
         --trainpref $mt_seg_path/train --validpref $mt_seg_path/valid --testpref $mt_seg_path/test \
         --destdir $mt_bin_path --joined-dictionary --srcdict $da_bin_path/dict.$slang.txt --workers 8

  # DONE-FLAG: flag done
  echo `date` > $done_file
fi

# 2. Train and test MT model
# DONE-FLAG: skip if it has been done, otherwise clear the folders
done_file=$mt_exp_path/done.$data-sent.train-augmodel
if [ -f $done_file ]; then
  echo `date`, Skip train-augmodel since it has been done on $(<$done_file).
else
  echo `date`, Training sent-level MT model ...
  bash -e scripts_gtrans/run-sent.sh $data train $mt_exp_path

  echo `date`, Test sent-level MT model ...
  bash -e scripts_gtrans/run-sent.sh $data test $mt_exp_path

  # DONE-FLAG: flag done
  echo `date` > $done_file
fi