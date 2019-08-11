import os
from typing import Dict, List, Tuple
from collections import defaultdict
from random import shuffle
import torch
import torch.nn as nn
import numpy as np

from wikiutil.metric import accuracy_nway, get_ranks
from wikiutil.property import read_subprop_file, get_pid2plabel, get_all_subtree, get_is_ancestor, \
    get_is_parent, get_leaves, read_nway_file, read_tbow_file
from wikiutil.util import load_tsv_as_dict, read_embeddings_from_text_file


class EmbModel(nn.Module):
    def __init__(self, emb, num_class, num_anc_class=0,
                 input_size=400, hidden_size=128, padding_idx=None, dropout=0.0,
                 only_prop=False, use_label=False,
                 vocab_size=None, tbow_emb_size=None, word_emb=None,
                 vocab_size2=None, tbow_emb_size2=None, word_emb2=None, only_tbow=False, use_weight=False):
        super(EmbModel, self).__init__()
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size
        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(emb).float(), freeze=True, padding_idx=padding_idx)
        self.only_prop = only_prop
        self.only_tbow = only_tbow
        self.num_anc_class = num_anc_class
        self.use_weight = use_weight
        if only_prop:
            input_size //= 2
        self.use_label = use_label
        if use_label:
            self.ff = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            )
            self.label_ff = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            )
        else:
            self.ff = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )
            self.pred_ff = nn.Linear(hidden_size, num_class)
            if num_anc_class:
                self.anc_pref_ff = nn.Linear(hidden_size, num_anc_class)
        if vocab_size:
            if word_emb is not None:
                self.tbow_emb = nn.Embedding.from_pretrained(
                    torch.tensor(word_emb).float(), freeze=True, padding_idx=padding_idx)
            else:
                self.tbow_emb = nn.Embedding(vocab_size, tbow_emb_size, padding_idx=padding_idx)
            self.tbow_key_ff = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_size - tbow_emb_size, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64)
            )
            self.tbow_value_ff = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(tbow_emb_size, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64)
            )
        if vocab_size2:
            if word_emb2 is not None:
                self.tbow_emb2 = nn.Embedding.from_pretrained(
                    torch.tensor(word_emb2).float(), freeze=True, padding_idx=padding_idx)
            else:
                self.tbow_emb2 = nn.Embedding(vocab_size2, tbow_emb_size2, padding_idx=padding_idx)



    def avg_emb(self, ind):
        # SHAPE: (batch_size, num_occs)
        ind_mask = ind.ne(self.padding_idx)
        ind_emb = self.emb(ind).sum(1) / ind_mask.sum(1).unsqueeze(-1).float()
        return ind_emb


    def combine_tbow_emb(self, lookup, tbow_ind, tbow_count, key_emb):
        # SHAPE: (batch_size, num_words)
        tbow_mask = tbow_ind.ne(self.padding_idx)
        tbow_count = tbow_count.float()
        tbow_count = tbow_count / (tbow_count.sum(-1, keepdim=True) + 1e-10)
        tbow_emb = lookup(tbow_ind)
        '''
        #tbow_emb = tbow_emb * nn.Sigmoid()(self.tbow_emb_gate(tbow_ind))
        bs, nw = tbow_emb.size()[:2]
        # SHAPE: (batch_size, emb_dim)
        key_emb =  self.tbow_key_ff(key_emb)
        # SHAPE: (batch_size, num_words, emb_dim)
        value_emb = self.tbow_value_ff(tbow_emb.view(bs * nw, -1)).view(bs, nw, -1)
        gate = nn.Sigmoid()((value_emb * key_emb.unsqueeze(1)).sum(-1))
        tbow_emb = tbow_emb * gate.unsqueeze(-1)
        '''
        if self.use_weight:
            tbow_emb = (tbow_emb * tbow_count.float().unsqueeze(-1)).sum(1)
        else:
            tbow_emb = tbow_emb.sum(1)
        return tbow_emb


    def forward(self, head, tail, labels,
                label_head=None, label_tail=None, tbow_ind=None, tbow_count=None, anc_labels=None,
                tbow_ind2=None, tbow_count2=None):
        # SHAPE: (batch_size, emb_dim)
        head_emb, tail_emb = self.avg_emb(head), self.avg_emb(tail)
        if self.use_label:
            # SHAPE: (num_class, emb_dim)
            label_head_emb, label_tail_emb = self.avg_emb(label_head), self.avg_emb(label_tail)

        if self.only_prop:
            emb = head_emb
            if self.use_label:
                label_emb = label_head_emb
        else:
            emb = torch.cat([head_emb, tail_emb], -1)
            if self.use_label:
                label_emb = torch.cat([label_head_emb, label_tail_emb], -1)

        if tbow_ind is not None and tbow_count is not None:
            tbow_emb = self.combine_tbow_emb(self.tbow_emb, tbow_ind, tbow_count, emb)
            if self.only_tbow:
                emb = tbow_emb
            else:
                emb = torch.cat([emb, tbow_emb], -1)

        if tbow_ind2 is not None and tbow_count2 is not None:
            tbow_emb = self.combine_tbow_emb(self.tbow_emb2, tbow_ind2, tbow_count2, emb)
            if self.only_tbow:
                emb = torch.cat([emb, tbow_emb], -1)
            else:
                emb = torch.cat([emb, tbow_emb], -1)

        pred_repr = self.ff(emb)

        if self.use_label:
            # SHAPE: (num_class, hidden_size)
            label_emb = self.label_ff(label_emb)
            # SHAPE: (batch_size, num_class)
            logits = torch.mm(pred_repr, label_emb.t())
        else:
            # SHAPE: (batch_size, num_class)
            logits = self.pred_ff(pred_repr)
            if self.num_anc_class:
                anc_logits = self.anc_pref_ff(pred_repr)

        loss = nn.CrossEntropyLoss()(logits, labels)
        if self.num_anc_class:
            anc_loss = nn.CrossEntropyLoss()(anc_logits, anc_labels)
            loss = loss + anc_loss
        return logits, loss


def get_prop_emb_ind(batch, emb_id2ind):
    return torch.tensor([[emb_id2ind[b[0][0]]] for b in batch]).long()


def get_occ_emb_ind(batch, emb_id2ind, num_occs, pos=0, batch_filter=None):
    batch2filter: Dict[int, set] = defaultdict(set)
    if batch_filter:
        for b in range(len(batch_filter)):
            for h, t in batch_filter[b][0][1]:
                batch2filter[b].add((h, t))
    return torch.tensor([[emb_id2ind[b[0][1][i][pos]]
                          if (i < len(b[0][1]) and tuple(b[0][1][i]) not in batch2filter[bid]) else 0
                          for i in range(num_occs)] for bid, b in enumerate(batch)]).long()


def get_tbow_ind(batch, max_num=0):
    word_ind_tensor = torch.tensor([[b[i][0] if i < len(b) else 0 for i in range(max_num)] for b in batch])
    word_count_tensor = torch.tensor([[b[i][1] if i < len(b) else 0 for i in range(max_num)] for b in batch])
    return word_ind_tensor, word_count_tensor


def data2tensor(batch, emb_id2ind, only_prop=False, num_occs=10, device=None,
                label_samples=None, use_tbow=0, use_tbow2=0, use_anc=False, num_occs_label=10):
    if use_tbow and not use_tbow2:
        batch, tbow_batch = zip(*batch)
    elif use_tbow2:
        batch, tbow_batch, tbow_batch2 = zip(*batch)

    label_head = label_tail = None
    if only_prop:
        head = get_prop_emb_ind(batch, emb_id2ind).to(device)
        tail = head
        if label_samples is not None:
            label_head = get_prop_emb_ind(label_samples, emb_id2ind).to(device)
            label_tail = label_head
    else:
        head = get_occ_emb_ind(batch, emb_id2ind, num_occs, pos=0).to(device)
        tail = get_occ_emb_ind(batch, emb_id2ind, num_occs, pos=1).to(device)
        if label_samples is not None:
            label_head = get_occ_emb_ind(
                label_samples, emb_id2ind, num_occs_label, pos=0, batch_filter=batch).to(device)
            label_tail = get_occ_emb_ind(
                label_samples, emb_id2ind, num_occs_label, pos=1, batch_filter=batch).to(device)

    if use_anc:
        labels = torch.tensor([b[1][0] for b in batch]).long().to(device)
        anc_labels = torch.tensor([b[1][1] for b in batch]).long().to(device)
    else:
        labels = torch.tensor([b[1] for b in batch]).long().to(device)
        anc_labels = None

    tbow_ind = tbow_count = None
    if use_tbow:
        tbow_ind, tbow_count = get_tbow_ind(tbow_batch, max_num=use_tbow)
        tbow_ind = tbow_ind.to(device)
        tbow_count = tbow_count.to(device)
    if use_tbow2:
        tbow_ind2, tbow_count2 = get_tbow_ind(tbow_batch2, max_num=use_tbow)
        tbow_ind2 = tbow_ind2.to(device)
        tbow_count2 = tbow_count2.to(device)
    else:
        tbow_ind2, tbow_count2 = None, None

    return head, tail, labels, label_head, label_tail, tbow_ind, tbow_count, anc_labels, tbow_ind2, tbow_count2


def samples2tensors(samples, batch_size, emb_id2ind,
                    num_occs, device, only_prop, label_samples=None,
                    use_tbow=0, use_tbow2=0, use_anc=False, num_occs_label=0):
    tensors: List = []
    for batch in range(0, len(samples), batch_size):
        batch = samples[batch:batch + batch_size]
        tensor = data2tensor(
            batch, emb_id2ind, only_prop=only_prop, num_occs=num_occs, device=device,
            label_samples=label_samples, use_tbow=use_tbow, use_tbow2=use_tbow2,
            use_anc=use_anc, num_occs_label=num_occs_label)
        tensors.append(tensor)
    return tensors


def one_epoch(model, optimizer, samples, tensors, batch_size, emb_id2ind,
              num_occs, device, only_prop, is_train, label_samples=None,
              use_tbow=0, use_anc=False, num_occs_label=0):
    if is_train:
        model.train()
        #shuffle(samples)
    else:
        model.eval()

    loss_li = []
    pred_li = []
    pred_label_li = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        tensor = tensors[i // batch_size]
        logits, loss = model(*tensor)
        labels = tensor[2]  # TODO: better structure
        if is_train and loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_li.append(loss.item())
        logits = logits.detach().cpu().numpy()
        if use_tbow:
            pred_li.extend([(b[0][0][0], logits[i], labels[i].item())
                            for i, b in enumerate(batch)])
        else:
            pred_li.extend([(b[0][0], logits[i], labels[i].item())
                            for i, b in enumerate(batch)])
        pred_label_li.extend([np.argmax(logits[i]) for i, b in enumerate(batch)])

    return loss_li, pred_li, pred_label_li


def run_emb_train(data_dir, emb_file, subprop_file, use_label=False, filter_leaves=False, only_test_on=None,
                  epoch=10, input_size=400, batch_size=128, use_cuda=False, early_stop=None,
                  num_occs=10, num_occs_label=10, hidden_size=128, dropout=0.0, lr=0.001, only_prop=False,
                  use_tbow=0, use_tbow2=0, tbow_emb_size=50, tbow_emb_size2=50, word_emb_file=None,
                  word_emb_file2=None, suffix='.tbow', suffix2='.tbow', only_tbow=False,
                  renew_word_emb=False, output_pred=False, use_ancestor=False, filter_labels=False,
                  acc_topk=1, use_weight=False, only_one_sample_per_prop=False, optimizer='adam'):
    subprops = read_subprop_file(subprop_file)
    pid2plabel = get_pid2plabel(subprops)
    subtrees, _ = get_all_subtree(subprops)
    is_parent = get_is_parent(subprops)
    is_ancestor = get_is_ancestor(subtrees)
    leaves = get_leaves(subtrees)
    pid2parent = dict([(c, p) for p, c in is_parent])

    emb_id2ind, emb = read_embeddings_from_text_file(
        emb_file, debug=False, emb_size=200, use_padding=True, first_line=False)

    train_samples = read_nway_file(os.path.join(data_dir, 'train.nway'))
    dev_samples = read_nway_file(os.path.join(data_dir, 'dev.nway'))
    test_samples = read_nway_file(os.path.join(data_dir, 'test.nway'))
    print('#train, #dev, #test {}, {}, {}'.format(len(train_samples), len(dev_samples), len(test_samples)))

    label2ind = load_tsv_as_dict(os.path.join(data_dir, 'label2ind.txt'), valuefunc=int)
    ind2label = dict((v, k) for k, v in label2ind.items())

    train_labels = set(s[1] for s in train_samples)
    test_labels = set(s[1] for s in test_samples)
    join_labels = train_labels & test_labels
    print('#labels in train & test {}'.format(len(join_labels)))

    anc2ind: Dict[str, int] = defaultdict(lambda: len(anc2ind))
    if use_ancestor:
        def get_anc_label(parent_label):
            parent_label = ind2label[parent_label]
            if parent_label in pid2parent:
                return anc2ind[pid2parent[parent_label]]
            else:
                return anc2ind['NO_ANC']

        train_samples = [((pid, poccs), (plabel, get_anc_label(plabel))) for (pid, poccs), plabel in train_samples]
        dev_samples = [((pid, poccs), (plabel, get_anc_label(plabel))) for (pid, poccs), plabel in dev_samples]
        test_samples = [((pid, poccs), (plabel, get_anc_label(plabel))) for (pid, poccs), plabel in test_samples]
    print('#ancestor {}'.format(len(anc2ind)))

    train_samples_tbow = dev_samples_tbow = test_samples_tbow = None
    vocab_size, vocab_size2 = None, None
    word_emb, word_emb2 = None, None
    if use_tbow:
        train_samples_tbow = read_tbow_file(os.path.join(data_dir, 'train' + suffix))
        assert len(train_samples_tbow) == len(train_samples)
        if use_tbow2:
            train_samples_tbow2 = read_tbow_file(os.path.join(data_dir, 'train' + suffix2))
            assert len(train_samples_tbow2) == len(train_samples)
            train_samples = list(zip(train_samples, train_samples_tbow, train_samples_tbow2))
        else:
            train_samples = list(zip(train_samples, train_samples_tbow))

        dev_samples_tbow = read_tbow_file(os.path.join(data_dir, 'dev' + suffix))
        assert len(dev_samples_tbow) == len(dev_samples)
        if use_tbow2:
            dev_samples_tbow2 = read_tbow_file(os.path.join(data_dir, 'dev' + suffix2))
            assert len(dev_samples_tbow2) == len(dev_samples)
            dev_samples = list(zip(dev_samples, dev_samples_tbow, dev_samples_tbow2))
        else:
            dev_samples = list(zip(dev_samples, dev_samples_tbow))

        test_samples_tbow = read_tbow_file(os.path.join(data_dir, 'test' + suffix))
        assert len(test_samples_tbow) == len(test_samples)
        if use_tbow2:
            test_samples_tbow2 = read_tbow_file(os.path.join(data_dir, 'test' + suffix2))
            assert len(test_samples_tbow2) == len(test_samples)
            test_samples = list(zip(test_samples, test_samples_tbow, test_samples_tbow2))
        else:
            test_samples = list(zip(test_samples, test_samples_tbow))

        if word_emb_file:
            word_emb_id2ind, word_emb = read_embeddings_from_text_file(
                word_emb_file, debug=False, emb_size=tbow_emb_size, first_line=False, use_padding=True, split_char=' ')
            vocab_size = len(word_emb_id2ind)
            if renew_word_emb:
                word_emb = None
        else:
            vocab_size = len(load_tsv_as_dict(os.path.join(data_dir, 'tbow.vocab')))

        if use_tbow2:
            if word_emb_file2:
                word_emb_id2ind2, word_emb2 = read_embeddings_from_text_file(
                    word_emb_file2, debug=False, emb_size=tbow_emb_size2, first_line=False, use_padding=True, split_char=' ')
                vocab_size2 = len(word_emb_id2ind2)
                if renew_word_emb:
                    word_emb2 = None
            else:
                vocab_size2 = len(load_tsv_as_dict(os.path.join(data_dir, 'tbow2.vocab')))

        print('vocab size 1 {}'.format(vocab_size))
        print('vocab size 2 {}'.format(vocab_size2))

    if filter_leaves:
        print('filter leaves')
        filter_pids = set(leaves)
        if use_tbow:
            train_samples = [s for s in train_samples if s[0][0][0] not in filter_pids]
            dev_samples = [s for s in dev_samples if s[0][0][0] not in filter_pids]
            test_samples = [s for s in test_samples if s[0][0][0] not in filter_pids]
        else:
            train_samples = [s for s in train_samples if s[0][0] not in filter_pids]
            dev_samples = [s for s in dev_samples if s[0][0] not in filter_pids]
            test_samples = [s for s in test_samples if s[0][0] not in filter_pids]

    if filter_labels:
        print('filter labels')
        if use_tbow:
            train_samples = [s for s in train_samples if s[0][1] in join_labels]
            dev_samples = [s for s in dev_samples if s[0][1] in join_labels]
            test_samples = [s for s in test_samples if s[0][1] in join_labels]
        else:
            train_samples = [s for s in train_samples if s[1] in join_labels]
            dev_samples = [s for s in dev_samples if s[1] in join_labels]
            test_samples = [s for s in test_samples if s[1] in join_labels]

    if only_one_sample_per_prop:
        def filter_first(samples, key_func):
            dup = set()
            new_samples = []
            for s in samples:
                k = key_func(s)
                if k in dup:
                    continue
                dup.add(k)
                new_samples.append(s)
            return new_samples

        if use_tbow:
            key_func = lambda s: s[0][0][0]
        else:
            key_func = lambda s: s[0][0]
        train_samples = filter_first(train_samples, key_func=key_func)
        dev_samples = filter_first(dev_samples, key_func=key_func)
        test_samples = filter_first(test_samples, key_func=key_func)

    if only_test_on:
        if use_tbow:
            test_samples = [s for s in test_samples if s[0][0][0] in only_test_on]
        else:
            test_samples = [s for s in test_samples if s[0][0] in only_test_on]

    if use_label:
        label_samples = read_nway_file(os.path.join(data_dir, 'label2occs.nway'))
        label_samples_dict: Dict[int, List] = defaultdict(list)
        for (pid, occs), l in label_samples:
            label_samples_dict[l].extend(occs)
        label_samples = [((ind2label[l], label_samples_dict[l]), l) for l in sorted(label_samples_dict.keys())]
    else:
        label_samples = None

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('#samples in train/dev/test: {} {} {}'.format(len(train_samples), len(dev_samples), len(test_samples)))

    emb_model = EmbModel(emb, len(label2ind), len(anc2ind), input_size=input_size,
                         hidden_size=hidden_size, padding_idx=0,
                         dropout=dropout, only_prop=only_prop,
                         use_label=use_label,
                         vocab_size=vocab_size, tbow_emb_size=tbow_emb_size, word_emb=word_emb,
                         vocab_size2=vocab_size2, tbow_emb_size2=tbow_emb_size2, word_emb2=word_emb2,
                         only_tbow=only_tbow, use_weight=use_weight)
    emb_model.to(device)
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(emb_model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(emb_model.parameters(), lr=lr)
    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(emb_model.parameters(), lr=lr)
    else:
        raise NotImplementedError

    last_metric, last_count = None, 0
    metrics = []
    train_tensors = samples2tensors(
        train_samples, batch_size, emb_id2ind, num_occs, device, only_prop, label_samples=label_samples,
        use_tbow=use_tbow, use_tbow2=use_tbow2, use_anc=use_ancestor, num_occs_label=num_occs_label)
    dev_tensors = samples2tensors(
        dev_samples, batch_size, emb_id2ind, num_occs, device, only_prop, label_samples=label_samples,
        use_tbow=use_tbow, use_tbow2=use_tbow2, use_anc=use_ancestor, num_occs_label=num_occs_label)
    test_tensors = samples2tensors(
        test_samples, batch_size, emb_id2ind, num_occs, device, only_prop, label_samples=label_samples,
        use_tbow=use_tbow, use_tbow2=use_tbow2, use_anc=use_ancestor, num_occs_label=num_occs_label)
    for e in range(epoch):
        # train
        train_loss, train_pred, _ = one_epoch(
            emb_model, optimizer, train_samples, train_tensors, batch_size,
            emb_id2ind, num_occs, device, only_prop, is_train=True,
            label_samples=label_samples, use_tbow=use_tbow, use_anc=use_ancestor,
            num_occs_label=num_occs_label)
        # dev
        dev_loss, dev_pred, _ = one_epoch(
            emb_model, optimizer, dev_samples, dev_tensors, batch_size,
            emb_id2ind, num_occs, device, only_prop, is_train=False,
            label_samples=label_samples, use_tbow=use_tbow, use_anc=use_ancestor,
            num_occs_label=num_occs_label)
        # test
        test_loss, test_pred, _ = one_epoch(
            emb_model, optimizer, test_samples, test_tensors, batch_size,
            emb_id2ind, num_occs, device, only_prop, is_train=False,
            label_samples=label_samples, use_tbow=use_tbow, use_anc=use_ancestor,
            num_occs_label=num_occs_label)

        train_metric, train_ranks, train_pred_label, _ = accuracy_nway(
            train_pred, ind2label=ind2label, topk=acc_topk, num_classes=len(label2ind))
        dev_metric, dev_ranks, dev_pred_label, _ = accuracy_nway(
            dev_pred, ind2label=ind2label, topk=acc_topk, num_classes=len(label2ind))
        test_metric, test_ranks, test_pred_label, _ = accuracy_nway(
            test_pred, ind2label=ind2label, topk=acc_topk, num_classes=len(label2ind))

        print('train: {:>.3f}, {:>.3f} dev: {:>.3f}, {:>.3f} test: {:>.3f}, {:>.3f}'.format(
            np.mean(train_loss), train_metric,
            np.mean(dev_loss), dev_metric,
            np.mean(test_loss), test_metric))

        if early_stop and last_metric and last_metric > dev_metric:
            last_count += 1
            if last_count >= early_stop:
                print('early stop')
                break
        last_metric = dev_metric
        metrics.append(test_metric)

    test_ranks = get_ranks(test_ranks, is_parent=is_parent, is_ancestor=is_ancestor)
    dev_ranks = get_ranks(dev_ranks, is_parent=is_parent, is_ancestor=is_ancestor)
    train_ranks = get_ranks(train_ranks, is_parent=is_parent, is_ancestor=is_ancestor)
    if output_pred:
        for fn in ['train', 'dev', 'test']:
            with open(os.path.join(data_dir, fn + '.pred'), 'w') as fout:
                for pl in eval('{}_pred_label'.format(fn)):
                    fout.write('{}\n'.format(pl))
    return metrics, test_ranks, dev_ranks, train_ranks
