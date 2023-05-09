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

data=$1
exp_path=$2

slang=en
tlang=de

# import data specific settings: max_len
source scripts_main/config-$data.sh

echo `date`, exp_path: $exp_path, data: $data, slang: $slang, tlang: $tlang

# DONE-FLAG: skip if it has been done, otherwise clear the folders
done_file=$exp_path/done.$data.prepare-finetune
if [ -f $done_file ]; then
  echo `date`, Skip prepare-finetune since it has been done on $(<$done_file).
else
  echo `date`, Prepraring data...
  mkdir -p $exp_path
  tok_path=$exp_path/$data.tokenized.$slang-$tlang
  seg_path_sent=$exp_path/$data-sent.segmented.$slang-$tlang
  seg_path_doc=$exp_path/$data-doc.segmented.$slang-$tlang
  bin_path_sent=$exp_path/$data-sent.binarized.$slang-$tlang
  bin_path_doc=$exp_path/$data-doc.binarized.$slang-$tlang
  rm $tok_path/* -rf
  rm $seg_path_sent/* -rf
  rm $seg_path_doc/* -rf
  rm $bin_path_sent/* -rf
  rm $bin_path_doc/* -rf

  # tokenize and sub-word
  bash exp_gtrans/prepare-bpe.sh raw_data/$data $tok_path

  # data builder
  python -m exp_gtrans.data_builder --datadir $tok_path --destdir $seg_path_sent/ --source-lang $slang --target-lang $tlang --max-tokens $max_len --max-sents 1
  python -m exp_gtrans.data_builder --datadir $tok_path --destdir $seg_path_doc/ --source-lang $slang --target-lang $tlang --max-tokens $max_len --max-sents 1000

  # Preprocess/binarize the data
  python -m fairseq_cli.preprocess --task translation_doc --source-lang $slang --target-lang $tlang \
         --trainpref $seg_path_sent/train --validpref $seg_path_sent/valid --testpref $seg_path_sent/test --destdir $bin_path_sent \
         --joined-dictionary --workers 8

  dict_path=$bin_path_sent/dict.$slang.txt
  python -m fairseq_cli.preprocess --task translation_doc --source-lang $slang --target-lang $tlang \
         --trainpref $seg_path_doc/train --validpref $seg_path_doc/valid --testpref $seg_path_doc/test --destdir $bin_path_doc \
         --srcdict $dict_path --tgtdict $dict_path --workers 8

  # DONE-FLAG: flag done
  echo `date` > $done_file
fi
