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
tok_path=$exp_path/$data.tokenized.$slang-$tlang
seg_path_sent=$exp_path/$data-sent.segmented.$slang-$tlang
seg_path_doc=$exp_path/$data-doc.segmented.$slang-$tlang
bin_path_sent=$exp_path/$data-sent.binarized.$slang-$tlang
bin_path_doc=$exp_path/$data-doc.binarized.$slang-$tlang

echo `date`, Prepraring data...

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
