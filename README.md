# Target-Side Data Augmentation

**This code is for ACL 2023 long paper "[Target-Side Augmentation for Document-Level Machine Translation](https://arxiv.org/abs/2305.04505)".**

**Python Version**: Python3.6

**Package Requirements**: torch==1.9.0

Before running the scripts, please make sure the submodule ./G-Trans is correctly cloned and setup the environment.
```
    bash setup.sh
```
(Notes: Our experiments are done on 4 GPUs of Tesla V100.)

## Main Experiments

Target-side augmentation for both sent-level Transformer and doc-level G-Transformer:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_tgtaug/run-all.sh nc2016 exp_main
```

The baseline for sent-level Transformer and doc-level G-Transformer:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_gtrans/run-baseline.sh nc2016 exp_main
```

## Back-translation + Targets-side Augmentation
Source-side augmentation with back-translation plus target-side augmentation with our DA model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_bothaug/run-all.sh nc2016 exp_backtrans
```

## Source-side Augmentation + Target-side Augmentation 
Source-side plus target-side augmentation with our DA model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_bothaug/run-all.sh nc2016 exp_ablation
```
