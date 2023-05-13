# Target-Side Data Augmentation

**This code is for ACL 2023 long paper "[Target-Side Augmentation for Document-Level Machine Translation](https://arxiv.org/abs/2305.04505)".**

**Python Version**: Python3.6

**Package Requirements**: torch==1.9.0

Before running the scripts, please make sure the submodule ./G-Trans is correctly cloned and setup the environment.
```
    bash setup.sh
```
(Notes: Our experiments are done on 4 GPUs of Tesla V100.)

### Main Experiments

* Target-side augmentation

Target-side augmentation for both sent-level Transformer and doc-level G-Transformer:
```
bash scripts_tgtaug/run-all.sh nc2016 exp_main
```

* Baseline

The baseline for sent-level Transformer and doc-level G-Transformer:
```
bash scripts_gtrans/run-baseline.sh nc2016 exp_main
```

### Additional Experiments

* Back-translation + Targets-side Augmentation

Source-side augmentation with back-translation plus target-side augmentation with our DA model:
```
bash scripts_bothaug/run-all.sh nc2016 exp_backtrans
```

* Source-side + Target-side Augmentation

Source-side plus target-side augmentation with our DA model:
```
bash scripts_bothaug/run-all.sh nc2016 exp_ablation
```

### Citation
```
@article{bao2023target,
  title={Target-Side Augmentation for Document-Level Machine Translation},
  author={Bao, Guangsheng and Teng, Zhiyang and Zhang, Yue},
  journal={arXiv preprint arXiv:2305.04505},
  year={2023}
}
```