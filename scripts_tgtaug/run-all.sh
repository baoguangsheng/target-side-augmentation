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

# 1. Prepare data
bash -e scripts_gtrans/prepare-finetune.sh $data $exp_path

# 2. Augment sent data
bash -e scripts_tgtaug/run-da-sent.sh $data $exp_path
bash -e scripts_tgtaug/run-mt-sent.sh $data $exp_path

# 3. Augment doc data
bash -e scripts_tgtaug/run-da-doc.sh $data $exp_path
bash -e scripts_tgtaug/run-mt-doc.sh $data $exp_path

# 4. Print BLEU scores
echo `date`, s-BLEU for sent Transformer:
cat $exp_path/subexp_mt/run-sent/test.$data-sent.$slang-$tlang.log | grep "Generate test with beam="

echo `date`, s/d-BLEU for G-Transformer(fnt.):
cat $exp_path/subexp_mt/run-finetune/test.$data-doc.$slang-$tlang.log | grep "Generate test with beam="

cd $cur_dir