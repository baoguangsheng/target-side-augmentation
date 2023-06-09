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

# import data specific settings: slang, tlang, dropout, max_len, patience_da, num_gpus
source $exp_path/scripts/config-$data.sh

doc_langs=$slang,$tlang
max_positions=$(($max_len * 2))

# 0. Prerequisite
from_seg_path=$exp_path/$data-sent.segmented.$slang-$tlang
from_bin_path=$exp_path/$data-sent.binarized.$slang-$tlang

# start DA experiment
da_exp_path=$exp_path/subexp_da
run_path=$da_exp_path/run-sent
mkdir -p $da_exp_path $run_path

# 1. Prepare DA dataset
# DONE-FLAG: skip if it has been done, otherwise clear the folders
done_file=$da_exp_path/done.$data-sent.prepare-da-data
if [ -f $done_file ]; then
  echo `date`, Skip prepare-da-data since it has been done on $(<$done_file).
else
  echo `date`, Prepare DA data for sent-level model ...
  da_seg_path=$da_exp_path/$data-sent.segmented.$slang-$tlang
  da_bin_path=$da_exp_path/$data-sent.binarized.$slang-$tlang
  mkdir -p $da_seg_path $da_bin_path

  rm $da_seg_path/* -rf
  rm $da_bin_path/* -rf

  python scripts_tgtaug/make_da_data.py --tok-path $from_seg_path --res-path $da_seg_path --slang $tlang --tlang $slang \
       --seed 1 --train-parts $num_gpus --repeat-times $repeat_times_sent $latent_args

  testpref=$da_seg_path/test
  for ((part=1; part<=$num_gpus; part++)); do
    testpref=$testpref,$da_seg_path/train-part${part}
  done

  python -m fairseq_cli.preprocess --task translation_doc --source-lang $tlang --target-lang $slang \
         --trainpref $da_seg_path/train --validpref $da_seg_path/valid --testpref $testpref \
         --destdir $da_bin_path --joined-dictionary --srcdict $from_bin_path/dict.$slang.txt --workers 8

  # DONE-FLAG: flag done
  echo `date` > $done_file
fi

# 2. Train DA model
# DONE-FLAG: skip if it has been done, otherwise clear the folders
done_file=$da_exp_path/done.$data-sent.train-da-model
if [ -f $done_file ]; then
  echo `date`, Skip train-da-model since it has been done on $(<$done_file).
else
  echo `date`, Training sent-level DA model ...
  da_bin_path=$da_exp_path/$data-sent.binarized.$slang-$tlang
  da_cp_path=$run_path/$data-sent.checkpoints.$slang-$tlang
  da_res_path=$run_path/$data-sent.results.$slang-$tlang
  mkdir -p $da_res_path

  python train.py $da_bin_path --save-dir $da_cp_path --tensorboard-logdir $da_cp_path --seed 444 --fp16 --num-workers 4 \
         --task translation_doc --source-lang $tlang --target-lang $slang --langs $doc_langs \
         --arch gtransformer_base --doc-mode full --share-all-embeddings \
         --optimizer adam --adam-betas "(0.9, 0.98)" --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --no-epoch-checkpoints \
         --dropout $dropout --max-source-positions $max_positions --max-target-positions $max_positions \
         --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience $patience_da \
         > $run_path/train.$data-sent.$slang-$tlang.log 2>&1

  echo `date`, Test sent-level DA model ...
  subset=test
  python -m fairseq_cli.generate $da_bin_path --path $da_cp_path/checkpoint_best.pt \
         --gen-subset $subset --batch-size 64 --beam 5 --max-len-a 1.2 --max-len-b 10 \
         --task translation_doc --source-lang $tlang --target-lang $slang --langs $doc_langs \
         --max-source-positions $max_positions --max-target-positions $max_positions \
         --doc-mode full --tokenizer moses --remove-bpe --sacrebleu \
         --gen-output $da_res_path/$subset > $run_path/$subset.$data-sent.$slang-$tlang.log 2>&1

  # DONE-FLAG: flag done
  echo `date` > $done_file
fi


# 3. Generate DA translations
# DONE-FLAG: skip if it has been done, otherwise clear the folders
done_file=$da_exp_path/done.$data-sent.generate-da-translations
if [ -f $done_file ]; then
  echo `date`, Skip generate-da-translations since it has been done on $(<$done_file).
else
  echo `date`, Generating sent-level DA translations ...
  da_bin_path=$da_exp_path/$data-sent.binarized.$slang-$tlang
  da_cp_path=$run_path/$data-sent.checkpoints.$slang-$tlang
  da_res_path=$run_path/$data-sent.results.$slang-$tlang

  # check readiness of GPUs
  python scripts_tgtaug/cuda_monitor.py --mode check --num-gpus $num_gpus

  WAIT_FOR_PIDS=""
  for ((part=1; part<=$num_gpus; part++)); do
    subset=test$part
    device=$(($part-1))
    CUDA_VISIBLE_DEVICES=$device nohup python -m fairseq_cli.generate $da_bin_path --path $da_cp_path/checkpoint_best.pt \
         --gen-subset $subset $sampling_args_sent \
         --task translation_doc --source-lang $tlang --target-lang $slang --langs $doc_langs \
         --max-source-positions $max_positions --max-target-positions $max_positions \
         --doc-mode full --tokenizer moses --remove-bpe --sacrebleu \
         --gen-output $da_res_path/$subset > $run_path/$subset.$data-sent.$slang-$tlang.log 2>&1 &
    WAIT_FOR_PIDS="$WAIT_FOR_PIDS $!"
  done

  # Wait for generate to complete
  wait $WAIT_FOR_PIDS
  echo `date`, Finish sent-level DA translation.

  # DONE-FLAG: flag doneg
  echo `date` > $done_file
fi