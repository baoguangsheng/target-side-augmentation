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

# 1. Run source-side augmentation
bash scripts_srcaug/run-all.sh $data $exp_path/exp_srcaug

# 2. Initialize the data for target-side augmentation
cp $exp_path/exp_srcaug/subexp_mt/$data-* $exp_path/exp_bothaug/. -rf
echo `date` > $exp_path/exp_bothaug/done.$data.prepare-finetune

# 3. Run target-side augmentation
bash scripts_tgtaug/run-all.sh $data $exp_path/exp_bothaug
