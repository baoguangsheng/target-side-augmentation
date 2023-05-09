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
from_seg_path=$exp_path/$data-doc.segmented.$slang-$tlang
from_bin_path=$exp_path/$data-doc.binarized.$slang-$tlang

# check DA experiment
da_exp_path=$exp_path/subexp_da
da_run_path=$da_exp_path/run-finetune
if [ ! -d $da_run_path ]; then
  echo `date`, The DA experiment is a prerequisite for MT experiment: $da_run_path
  exit -1
fi

# start MT experiment
mt_exp_path=$exp_path/subexp_mt
mkdir -p $mt_exp_path

# check sent MT model
sent_model=$mt_exp_path/run-sent/$data-sent.checkpoints.$slang-$tlang/checkpoint_best.pt
if [ ! -f $sent_model ]; then
  echo `date`, The sent MT model is required: $sent_model
  exit -1
fi

# 1. Prepare dataset with augmented data
# DONE-FLAG: skip if it has been done, otherwise clear the folders
done_file=$mt_exp_path/done.$data-doc.prepare-mt-data
if [ -f $done_file ]; then
  echo `date`, Skip prepare-mt-data since it has been done on $(<$done_file).
else
  echo `date`, Prepare MT data for doc-level model...
  da_seg_path=$da_exp_path/$data-doc.segmented.$slang-$tlang
  da_bin_path=$da_exp_path/$data-doc.binarized.$slang-$tlang
  da_res_path=$da_run_path/$data-doc.results.$slang-$tlang

  mt_seg_path=$mt_exp_path/$data-doc.segmented.$slang-$tlang
  mt_bin_path=$mt_exp_path/$data-doc.binarized.$slang-$tlang
  mkdir -p $mt_seg_path $mt_bin_path

  rm $mt_seg_path/* -rf
  rm $mt_bin_path/* -rf

  cp $from_seg_path/* $mt_seg_path/. -rf
  for ((part=1; part<=$num_gpus; part++)); do
    trainpart=train-part$part
    subset=test$part
    cat $da_seg_path/$trainpart.$slang.origion >> $mt_seg_path/train.$slang
    cat $da_res_path/$subset.seg.gen >> $mt_seg_path/train.$tlang
  done

  python -m fairseq_cli.preprocess --task translation_doc --source-lang $slang --target-lang $tlang \
         --trainpref $mt_seg_path/train --validpref $mt_seg_path/valid --testpref $mt_seg_path/test \
         --destdir $mt_bin_path --joined-dictionary --srcdict $da_bin_path/dict.$slang.txt --workers 8

  # DONE-FLAG: flag done
  echo `date` > $done_file
fi

# 2. Train and test MT model
# DONE-FLAG: skip if it has been done, otherwise clear the folders
done_file=$mt_exp_path/done.$data-doc.train-mt-model
if [ -f $done_file ]; then
  echo `date`, Skip train-mt-model since it has been done on $(<$done_file).
else
  echo `date`, Training doc-level MT model with sent model from $sent_model ...
  bash -e scripts_gtrans/run-finetune.sh $data train $mt_exp_path

  echo `date`, Test doc-level MT model ...
  bash -e scripts_gtrans/run-finetune.sh $data test $mt_exp_path

  # DONE-FLAG: flag done
  echo `date` > $done_file
fi
