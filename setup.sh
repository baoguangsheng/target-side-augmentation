#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

pip install protobuf==3.19.1
pip install ninja==1.11.1 sacrebleu==1.4.14 tensorboard==2.6.0
pip install --editable ./G-Trans/.

# create link folders in G-Trans for convenience to run the scripts
if [ ! -d "./G-Trans/scripts_gtrans" ]; then
  cur_dir=$(pwd)
  ln -s $cur_dir/scripts_gtrans ./G-Trans/scripts_gtrans
  ln -s $cur_dir/scripts_srcaug ./G-Trans/scripts_srcaug
  ln -s $cur_dir/scripts_tgtaug ./G-Trans/scripts_tgtaug
fi
