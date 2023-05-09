#!/usr/bin/env python3 -u

import os
import os.path as path
import numpy as np


class ObserveRatio:
    def __init__(self, expression):
        assert expression.startswith('Beta(') and expression.endswith(')')
        self.values = literal_eval(expression[5:-1])
        assert len(self.values) == 2

    def sample(self):
        return np.random.beta(self.values[0], self.values[1])

    def __str__(self):
        arg = ','.join(map(str, self.values))
        return f'Beta({arg})'

    def __repr__(self):
        return str(self)


def save_lines(file_name, lines):
    import codecs
    with codecs.open(file_name, 'w', 'utf-8') as fout:
        for line in lines:
            print(line, file=fout)


def load_lines(file_name):
    import codecs
    with codecs.open(file_name, 'r', 'utf-8') as fin:
        lines = [line.strip() for line in fin]
    return lines


def _sample_ngrams(args, tgt_sent):
    tgt_sent = tgt_sent[1:-1]  # exclude <s> and </s>
    # generate n-grams
    tgt_ngrams = []
    if args.observe_replacement:  # with replacement
        total = 0
        while total < len(tgt_sent):
            start = np.random.randint(0, len(tgt_sent))
            n = np.random.randint(args.observe_ngram[0], args.observe_ngram[1] + 1)
            n = min(n, len(tgt_sent) - start, len(tgt_sent) - total)
            tgt_ngrams.append(tgt_sent[start: start + n])
            total += n
    else:  # without replacement
        start = 0
        while start < len(tgt_sent):
            n = np.random.randint(args.observe_ngram[0], args.observe_ngram[1] + 1)
            tgt_ngrams.append(tgt_sent[start: start + n])
            start += n
    # select a ratio of tokens
    np.random.shuffle(tgt_ngrams)
    tgt_ngrams = sum(tgt_ngrams, [])
    assert len(tgt_ngrams) == len(tgt_sent)
    obs_ratio = args.observe_ratio.sample()
    obs_len = round(obs_ratio * len(tgt_sent))
    obs_len = np.clip(obs_len, args.observe_minlen, len(tgt_sent))
    return tgt_ngrams[: obs_len]


def extend_src_with_latent(args, src, tgt):
    src_sents = src.replace('</s> <s>', '</s> ##12321## <s>').split(' ##12321## ')
    tgt_sents = tgt.replace('</s> <s>', '</s> ##12321## <s>').split(' ##12321## ')
    assert len(src_sents) == len(tgt_sents)
    src_words = [sent.split() for sent in src_sents]
    tgt_words = [sent.split() for sent in tgt_sents]
    new_lines = []
    if args.observe:
        for cur in range(args.repeat_times):
            newsrc = []
            for src_sent, tgt_sent in zip(src_words, tgt_words):
                observe = [args.observe_token]
                observe.extend(_sample_ngrams(args, tgt_sent))
                newsrc.append(' '.join(src_sent[:-1] + observe + src_sent[-1:]))
            newsrc = ' '.join(newsrc)
            new_lines.append((newsrc, tgt, src))
    else:  # without "<obs> ..."
        for _ in range(args.repeat_times):
            new_lines.append((src, tgt, src))

    return new_lines


def make_da_data(args):
    if not path.exists(args.res_path):
        os.mkdir(args.res_path)

    np.random.seed(args.seed)
    for corpus in ['test', 'valid', 'train']:
        src_file = path.join(args.tok_path, f'{corpus}.{args.slang}')
        tgt_file = path.join(args.tok_path, f'{corpus}.{args.tlang}')
        src_lines = load_lines(src_file)
        tgt_lines = load_lines(tgt_file)
        assert len(src_lines) == len(tgt_lines)
        new_lines = []
        for src, tgt in zip(src_lines, tgt_lines):
            new_lines.extend(extend_src_with_latent(args, src, tgt))
        src_file = path.join(args.res_path, f'{corpus}.{args.slang}')
        tgt_file = path.join(args.res_path, f'{corpus}.{args.tlang}')
        src_origion_file = path.join(args.res_path, f'{corpus}.{args.slang}.origion')
        save_lines(src_file, [src for src, _, _ in new_lines])
        save_lines(tgt_file, [tgt for _, tgt, _ in new_lines])
        save_lines(src_origion_file, [src_ori for _, _, src_ori in new_lines])
        print('File saved %s' % src_file)
        print('File saved %s' % tgt_file)
        print('File saved %s' % src_origion_file)
        if corpus == 'train':
            # split train set into 4 parts so that we can generate DA translations on 4 GPUs parallel
            part_size = len(new_lines) // args.train_parts + 1
            for part in range(1, 1 + args.train_parts):
                part_lines = new_lines[(part - 1) * part_size: part * part_size]
                src_file = path.join(args.res_path, f'{corpus}-part{part}.{args.slang}')
                tgt_file = path.join(args.res_path, f'{corpus}-part{part}.{args.tlang}')
                src_origion_file = path.join(args.res_path, f'{corpus}-part{part}.{args.slang}.origion')
                save_lines(src_file, [src for src, _, _ in part_lines])
                save_lines(tgt_file, [tgt for _, tgt, _ in part_lines])
                save_lines(src_origion_file, [src_ori for _, _, src_ori in part_lines])
                print('File saved %s' % src_file)
                print('File saved %s' % tgt_file)
                print('File saved %s' % src_origion_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok-path", default='exp_data/nc2016-doc.segmented.en-de')
    parser.add_argument("--res-path", default='exp_test/nc2016-doc.segmented.en-de')
    parser.add_argument("--slang", default='en')
    parser.add_argument("--tlang", default='de')
    parser.add_argument("--train-parts", type=int, default=4)
    parser.add_argument("--repeat-times", type=int, default=9)
    parser.add_argument("--observe", action='store_true', default=False)
    parser.add_argument("--observe-token", default='<obs>')
    parser.add_argument("--observe-ratio", default='Beta(2,3)')
    parser.add_argument("--observe-ngram", default='[1,3]')
    parser.add_argument("--observe-minlen", type=int, default=0)
    parser.add_argument("--observe-replacement", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    from ast import literal_eval
    args.observe_ratio = ObserveRatio(args.observe_ratio)
    args.observe_ngram = literal_eval(args.observe_ngram)
    print(args)

    make_da_data(args)
