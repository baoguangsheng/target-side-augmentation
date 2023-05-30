#!/usr/bin/env python3 -u

import io
import os
import os.path as path
import random
from fairseq.data import encoders
from fairseq import bleu
from tqdm import tqdm
import numpy as np
from .make_da_data import load_lines, save_lines

def remove_seps(text):
    sents = [s.strip() for s in text.replace('<s>', '').split('</s>')]
    sents = [s for s in sents if len(s) > 0]
    return sents

def load_aug_data(args):
    src_file = path.join(args.aug_path, f'train.{args.slang}')
    tgt_file = path.join(args.aug_path, f'train.{args.tlang}')

    src_lines = load_lines(src_file)
    tgt_lines = load_lines(tgt_file)
    assert len(src_lines) == len(tgt_lines)

    ncopies = args.aug_num + 1
    assert len(src_lines) % ncopies == 0

    copylen = len(src_lines) // ncopies
    parallels = [(src, [tgt]) for src, tgt in zip(src_lines[:copylen], tgt_lines[:copylen])]
    for i in tqdm(range(copylen)):
        src = parallels[i][0]
        tgts = parallels[i][1]
        for j in range(args.aug_num):
            aug_src = src_lines[copylen + i * args.aug_num + j]
            aug_tgt = tgt_lines[copylen + i * args.aug_num + j]
            assert aug_src == src
            tgts.append(aug_tgt)
    return parallels

def convert_to_sentences(parallels):
    parals_sent = []
    for src, tgts in parallels:
        src = src.replace('</s> <s>', '</s> ##12321## <s>').split(' ##12321## ')
        tgts = [tgt.replace('</s> <s>', '</s> ##12321## <s>').split(' ##12321## ') for tgt in tgts]
        assert all(len(tgt) == len(src) for tgt in tgts)
        tgts = list(zip(*tgts))
        parals_sent.extend(zip(src, tgts))
    return parals_sent

class TranslationMetric:
    def __init__(self, args):
        args.tokenizer = 'moses'
        args.source_lang = args.slang
        args.target_lang = args.tlang
        self.tokenizer = encoders.build_tokenizer(args)
        self.scorer = bleu.SacrebleuScorer()

    def detok(self, text):
        return ' '.join(remove_seps(self.tokenizer.decode(text.replace('@@ ', ''))))

    def deviation(self, ref, gen):
        self.scorer.reset()
        self.scorer.add_string(ref, gen)
        res = 100 - self.scorer.score()
        return res

    def diversity(self, gens):
        devis = []
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                devis.append(self.deviation(gens[i], gens[j]))
        return np.mean(devis)

def eval_aug_deviation(args):
    # load data
    parallels = load_aug_data(args)
    parallels = convert_to_sentences(parallels)

    random.seed(1)
    random.shuffle(parallels)
    parallels = parallels[:20000]  # estimate using random 2w samples

    # get metrics
    metric = TranslationMetric(args)
    devis = []
    for src, tgts in tqdm(parallels):
        tgts = [metric.detok(tgt) for tgt in tgts]
        tgt = tgts[0]
        gens = tgts[1:]
        for gen in gens:
            devis.append(metric.deviation(tgt, gen))
    print(f'From {args.suffix}, mean of deviation: {np.mean(devis):.2f}')
    return devis

def eval_aug_diverse(args):
    # load data
    parallels = load_aug_data(args)
    parallels = convert_to_sentences(parallels)

    random.seed(1)
    random.shuffle(parallels)
    parallels = parallels[:20000]  # estimate using random 2w samples

    # get metrics
    metric = TranslationMetric(args)
    divs = []
    for src, tgts in tqdm(parallels):
        tgts = [metric.detok(tgt) for tgt in tgts]
        gens = tgts[1:]
        divs.append(metric.diversity(gens))
    print(f'From {args.suffix}, mean of diversity: {np.mean(divs):.2f}')
    return divs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug-path", default='exp_main/subexp_mt/nc2016-doc.segmented.en-de')
    parser.add_argument("--slang", default='en')
    parser.add_argument("--tlang", default='de')
    parser.add_argument("--aug-num", default=9, help='augmented times of the dataset')
    args = parser.parse_args()

    # Evaluate the Devication of the DA generated translations
    eval_aug_deviation(args)

    # Evaluate the Diversity of the DA generated translations
    eval_aug_diverse(args)
