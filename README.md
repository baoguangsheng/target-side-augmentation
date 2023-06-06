# Target-Side Data Augmentation

**This code is for ACL 2023 long paper "Target-Side Augmentation for Document-Level Machine Translation".**

[Paper](https://arxiv.org/pdf/2305.04505v2.pdf) 
| [Poster](https://github.com/baoguangsheng/target-side-augmentation/blob/main/docs/poster.pdf)
| [Slides](https://github.com/baoguangsheng/target-side-augmentation/blob/main/docs/slides.pdf)
| [Video](https://www.youtube.com/watch?v=3PRPBcoRKnw)

## Brief Intro
![Brief intro of target-side data augmentation.](https://github.com/baoguangsheng/target-side-augmentation/blob/main/docs/introduction.png)


## Environment
* Python3.6
* PyTorch1.9.0
* Setup the environment:
  ```
    git clone --recursive https://github.com/baoguangsheng/target-side-augmentation.git
    cd target-side-augmentation
    bash setup.sh
  ```
(Notes: our experiments are run on 4 GPUs of Tesla V100.)


## Description of Codes
* ./G-Trans -> submodule reference to G-Transformer.
* ./scripts_gtrans -> updated G-Transformer scripts with new settings.
* ./scripts_tgtaug -> scripts for target-side augmentation.
* ./scripts_srcaug -> scripts for source-side augmentation.
* ./scripts_bothaug -> integrated scripts for both-side augmentation.

## Experiment Workspace
Following folders are created for our experiments:
* ./exp_main -> main experiments for target-side augmentation.
* ./exp_main/subexp_da -> for DA model.
* ./exp_main/subexp_mt -> for MT model.
* ./exp_backtrans -> back-translation plus target-side augmentation.
* ./exp_backtrans/exp_srcaug -> source-side aug with back-translation.
* ./exp_backtrans/exp_bothaug -> further target-side aug with DA model.

(Notes: we put recent experiment logs in ./logs/*.zip for reference.)

## How to Run 

### Baselines + Target-side augmentation

The scripts will run four steps: 
1) Prepare data; 
2) Sent-level augmentation on Transformer; 
3) Doc-level augmentation on G-Transformer; 
4) Report s/d-BLEU scores. 
```
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_tgtaug/run-all.sh nc2016 exp_main
```
The "nc2016" is the data name for News, which could be replaced with "iwslt17" for TED and "europarl7" for Europarl.

We place the configuration scripts for the experiment at exp_main/scripts. After running, two sub-folders will be created. One for the DA model (exp_main/subexp_da), and another for the MT model (exp_main/subexp_mt).


### Back-translation + Targets-side Augmentation

  The scripts will run three steps: 1) Run source-side augmentation with back-translation; 2) Initialize the data with back-translated sources for target-side augmentation; 3) Run target-side augmentation with DA model. 
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_bothaug/run-all.sh nc2016 exp_backtrans
  ```
  We treat back-translation as a special DA model with specific configuration (exp_backtrans/exp_srcaug/scripts). We put experiments with back-translation under exp_backtrans/exp_srcaug and experiments with further target-side augmentation under exp_backtrans/exp_bothaug. 


### Transformer and G-Transformer baselines

  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_gtrans/run-baseline.sh nc2016 exp_main
  ```
  We put the experiments under exp_main/subexp_baseline.

### Citation

```
@article{bao2023target,
  title={Target-Side Augmentation for Document-Level Machine Translation},
  author={Bao, Guangsheng and Teng, Zhiyang and Zhang, Yue},
  journal={arXiv preprint arXiv:2305.04505},
  year={2023}
}
```