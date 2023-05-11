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
bin_path=$exp_path/$data-doc.binarized.$slang-$tlang

run_path=$exp_path/run-finetune
mkdir -p $run_path
echo `date`, run path: $run_path

cp_path=$run_path/$data-doc.checkpoints.$slang-$tlang
res_path=$run_path/$data-doc.results.$slang-$tlang
doc_langs=$slang,$tlang
max_positions=$(($max_len * 2))

if [ $mode == "train" ]; then
  echo `date`, Training document-level model...
  sent_model=$exp_path/run-sent/$data-sent.checkpoints.$slang-$tlang/checkpoint_best.pt

  echo `date`, Training doc model with sent model from $sent_model ...
  python train.py $bin_path --save-dir $cp_path --tensorboard-logdir $cp_path --seed 444 --num-workers 4 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
         --arch gtransformer_base --doc-mode partial --share-all-embeddings \
         --optimizer adam --adam-betas "(0.9, 0.98)" \
         --lr-scheduler inverse_sqrt --lr 5e-04 --warmup-updates 4000 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --no-epoch-checkpoints \
         --dropout $dropout --max-source-positions $max_positions --max-target-positions $max_positions \
         --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience $patience_mt \
         --restore-file $sent_model --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
         --load-partial --doc-double-lr --lr-scale-pretrained 0.2 \
         --encoder-ctxlayers 2 --decoder-ctxlayers 2 --cross-ctxlayers 2 \
         --doc-noise-mask 0.1 --doc-noise-epochs 30 > $run_path/train.$data-doc.$slang-$tlang.log 2>&1
elif [ $mode == "test" ]; then
  mkdir -p $res_path
  subset=test

  echo `date`, Testing model on $subset dataset...
  python -m fairseq_cli.generate $bin_path --path $cp_path/checkpoint_best.pt \
         --gen-subset $subset --batch-size 8 --beam 5 --max-len-a 1.2 --max-len-b 10 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
         --max-source-positions $max_positions --max-target-positions $max_positions \
         --doc-mode partial --tokenizer moses --remove-bpe --sacrebleu \
         --gen-output $res_path/$subset > $run_path/$subset.$data-doc.$slang-$tlang.log 2>&1
else
  echo Unknown mode ${mode}.
fi