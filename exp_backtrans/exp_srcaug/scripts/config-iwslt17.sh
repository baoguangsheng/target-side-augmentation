#!/usr/bin/env bash

# config variables
slang=en
tlang=de

dropout=0.3
max_len=1024
patience_da=5
patience_mt=10
num_gpus=4

repeat_times_sent=1
repeat_times_doc=1
latent_args=""

sampling_args_sent="--batch-size 64 --beam 5 --max-len-a 1.2 --max-len-b 10"
sampling_args_doc="--batch-size 8 --beam 5 --max-len-a 1.2 --max-len-b 10"

echo `date`, Config for IWSLT17: dropout=$dropout, max_len=$max_len, patience_da=$patience_da, patience_mt=$patience_mt, num_gpus=$num_gpus
echo `date`,                    repeat_times_sent=$repeat_times_sent, repeat_times_doc=$repeat_times_doc, latent_args=$latent_args
echo `date`,                    sampling_args_sent=$sampling_args_sent
echo `date`,                    sampling_args_doc=$sampling_args_doc
