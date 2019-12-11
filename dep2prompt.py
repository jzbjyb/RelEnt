from typing import List, Tuple, Dict
import argparse
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json
import string
import spacy

nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words
punct = set(list(string.punctuation))


def check_prompt(prompt: List[str]): 
    if np.all([w in stopwords or w in punct or w in {'[X]', '[Y]'} for w in prompt]):
        return False
    return True


def get_prompt(sent_line: str, dep_line: str, sub_label: str = '[X]', obj_label: str = '[Y]'):
    hid, tid, hpos, tpos, sent = sent_line.split('\t')
    hid_, tid_, dep = dep_line.split('\t')
    if hid != hid_ or tid != tid_:
        raise Exception('not the same sentence')
    sent: List[str] = sent.split(' ')
    dep_path: List[str] = dep.split(' ')
    hs, he = list(map(int, hpos.split(':')))
    ts, te = list(map(int, tpos.split(':')))

    leftmost_li = [hs, ts]  # inclusive
    rightmost_li = [he, te]  # exclusive

    for i, dep in enumerate(dep_path):
        if i % 2 == 0:  # skip dep label
            continue
        inds = [j for j, x in enumerate(sent) if x == dep]
        ind = sorted(inds, key=lambda x: abs(x - hs))[0]
        leftmost_li.append(ind)
        rightmost_li.append(ind + 1)

    leftmost = np.min(leftmost_li)
    rightmost = np.max(rightmost_li)

    for i in range(hs, he):
        sent[i] = sub_label
    for i in range(ts, te):
        sent[i] = obj_label

    prompts = sent[leftmost:rightmost]
    sub_met, obj_met = False, False
    filter_prompts = []
    for w in prompts:
        if w == sub_label:
            if sub_met:
                continue
            sub_met = True
        if w == obj_label:
            if obj_met:
                continue
            obj_met = True
        filter_prompts.append(w)

    if len(filter_prompts) >= 20:
        return None
    if not check_prompt(filter_prompts):
        return None

    return ' '.join(filter_prompts)


def check_dep(prompt: str, sub_label: str = '[X]', obj_label: str = '[Y]'):
    prompt = prompt.rstrip('.').strip()
    if (prompt.startswith(sub_label) or prompt.startswith(obj_label)) \
        and (prompt.endswith(sub_label) or prompt.endswith(obj_label)):
        return False
    for w in prompt.split(' '):
        if w[0].isupper():
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser('convert dependencies to prompts')
    parser.add_argument('--sent_dir', type=str, required=True, help='directory to sentences')
    parser.add_argument('--out_dir', type=str, required=True, help='output file')
    parser.add_argument('--only_dep_dir', type=str, help='prompts that can only be generated by dep', default=None)
    args = parser.parse_args()

    if args.only_dep_dir:
        for root, dirs, files in os.walk(args.out_dir):
            for file in files:
                with open(os.path.join(args.out_dir, file), 'r') as fin, \
                    open(os.path.join(args.only_dep_dir, file), 'w') as fout:
                    for l in fin:
                        l = json.loads(l)
                        if l['wikipedia_count'] < 10:
                            continue
                        if not check_dep(l['template']):
                            continue
                        fout.write(json.dumps(l) + '\n')
        exit(0)

    pid2prompts = defaultdict(lambda: defaultdict(lambda: 0))

    for dir_ind in range(10):
        for root, dirs, files in os.walk(os.path.join(args.sent_dir, '{}'.format(dir_ind))):
            for file in tqdm(files):
                sent_file = os.path.join(root, file)
                dep_file = os.path.join(os.path.join(args.sent_dir, '{}_dep'.format(dir_ind)), file)
                pid = file.split('.')[0].split('_')[0]
                sent_file = open(sent_file, 'r')
                dep_file = open(dep_file, 'r')
                while True:
                    sent = sent_file.readline().rstrip('\n')
                    dep = dep_file.readline().rstrip('\n')
                    if sent == '' or dep == '':
                        break
                    prompt = get_prompt(sent, dep)
                    if prompt is not None:
                        pid2prompts[pid][prompt] += 1
                    #print(prompt)
                    #print(dep.split('\t')[-1])
                    #input()
                sent_file.close()
                dep_file.close()
        for pid, prompts in pid2prompts.items():
            prompts = sorted(prompts.items(), key=lambda x: -x[1])
            with open(os.path.join(args.out_dir, pid + '.jsonl'), 'w') as fout:
                for i, (prompt, count) in enumerate(prompts):
                    if i >= 100:
                        break
                    fout.write(json.dumps({'relation': pid, 'template': prompt + ' .', 'wikipedia_count': count}) + '\n')
