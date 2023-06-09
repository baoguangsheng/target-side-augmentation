#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17 train exp_test"
    exit
fi

# run command
data=$1
mode=$2
exp_path=$3


# import data specific settings: slang, tlang, dropout, max_len, patience_mt
source $exp_path/../scripts/config-$data.sh

echo `date`, data: $data, mode: $mode, exp_path: $exp_path, slang: $slang, tlang: $tlang
bin_path=$exp_path/$data-sent.binarized.$slang-$tlang

run_path=$exp_path/run-sent
mkdir -p $run_path
echo `date`, run path: $run_path

cp_path=$run_path/$data-sent.checkpoints.$slang-$tlang
res_path=$run_path/$data-sent.results.$slang-$tlang
doc_langs=$slang,$tlang
max_positions=$(($max_len * 2))

if [ $mode == "train" ]; then
  echo `date`, Training sentence-level model...
  # G-Transformer with the doc-mode of 'full' is the same as Transformer but with different naming on the attention modules.
  python train.py $bin_path --save-dir $cp_path --tensorboard-logdir $cp_path --seed 444 --fp16 --num-workers 4 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
         --arch gtransformer_base --doc-mode full --share-all-embeddings \
         --optimizer adam --adam-betas "(0.9, 0.98)" --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --no-epoch-checkpoints \
         --dropout $dropout --max-source-positions $max_positions --max-target-positions $max_positions \
         --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience $patience_mt \
         > $run_path/train.$data-sent.$slang-$tlang.log 2>&1

elif [ $mode == "test" ]; then
  mkdir -p $res_path
  subset=test

  echo `date`, Testing model on $subset dataset...
  python -m fairseq_cli.generate $bin_path --path $cp_path/checkpoint_best.pt \
         --gen-subset $subset --batch-size 64 --beam 5 --max-len-a 1.2 --max-len-b 10 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
         --max-source-positions $max_positions --max-target-positions $max_positions\
         --doc-mode full --tokenizer moses --remove-bpe --sacrebleu \
         --gen-output $res_path/$subset > $run_path/$subset.$data-sent.$slang-$tlang.log 2>&1
else
  echo Unknown mode ${mode}.
fi