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

# setup the environment
set -e  # exit if error
umask 002  # avoid root privilege in docker

cur_dir=$(pwd)
exp_path=$cur_dir/$exp_path
cd ./G-Trans

# start baseline experiment
baseline_exp_path=$exp_path/subexp_baseline
mkdir -p $baseline_exp_path

# 1. Prepare data
bash -e scripts_gtrans/prepare-finetune.sh $data $exp_path
cp $exp_path/*${data}* $baseline_exp_path/. -rf

# 2. Train and test sent MT model
# DONE-FLAG: skip if it has been done, otherwise clear the folders
done_file=$baseline_exp_path/done.$data-sent.train-mt-model
if [ -f $done_file ]; then
  echo `date`, Skip train-mt-model since it has been done on $(<$done_file).
else
  echo `date`, Training sent-level MT model ...
  bash -e scripts_gtrans/run-sent.sh $data train $baseline_exp_path

  echo `date`, Test sent-level MT model ...
  bash -e scripts_gtrans/run-sent.sh $data test $baseline_exp_path

  # DONE-FLAG: flag done
  echo `date` > $done_file
fi

# 3. Train and test doc MT model
# DONE-FLAG: skip if it has been done, otherwise clear the folders
done_file=$baseline_exp_path/done.$data-doc.train-mt-model
if [ -f $done_file ]; then
  echo `date`, Skip train-mt-model since it has been done on $(<$done_file).
else
  echo `date`, Training doc-level MT model with sent model from $sent_model ...
  bash -e scripts_gtrans/run-finetune.sh $data train $baseline_exp_path

  echo `date`, Test doc-level MT model ...
  bash -e scripts_gtrans/run-finetune.sh $data test $baseline_exp_path

  # DONE-FLAG: flag done
  echo `date` > $done_file
fi

cd $cur_dir



