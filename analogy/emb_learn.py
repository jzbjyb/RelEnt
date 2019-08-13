import os
from typing import Dict, List, Tuple
from collections import defaultdict
from random import shuffle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from wikiutil.metric import accuracy_nway, get_ranks
from wikiutil.property import read_subprop_file, get_pid2plabel, get_all_subtree, get_is_ancestor, \
    get_is_parent, get_leaves, read_nway_file, read_tbow_file, read_sent_file
from wikiutil.util import load_tsv_as_dict, read_embeddings_from_text_file
from .emb_gnn import build_graph, EmbGnnModel


class EmbModel(nn.Module):
    def __init__(self, emb, num_class, num_anc_class=0,
                 input_size=400, hidden_size=128, padding_idx=None, dropout=0.0,
                 only_prop=False, use_label=False,
                 vocab_size=None, tbow_emb_size=None, word_emb=None,
                 vocab_size2=None, tbow_emb_size2=None, word_emb2=None,
                 sent_vocab_size=None, sent_emb_size=None, sent_emb=None,
                 only_tbow=False, use_weight=False, sent_emb_method='rnn'):
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
        assert sent_emb_method in {'avg', 'rnn', 'cnn'}
        self.sent_emb_method = sent_emb_method

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

        if sent_vocab_size:
            if sent_emb is not None:
                self.sent_emb = nn.Embedding.from_pretrained(
                    torch.tensor(sent_emb).float(), freeze=False, padding_idx=padding_idx)
            else:
                self.sent_emb = nn.Embedding(sent_vocab_size, sent_emb_size, padding_idx=padding_idx)

            if sent_emb_method == 'rnn':
                self.num_rnn_layer = 1
                self.num_rnn_direction = 1
                self.rnn_hidden_size = 64  # TODO: add param
                self.rnn = nn.LSTM(sent_emb_size, self.rnn_hidden_size,
                                   self.num_rnn_layer, bidirectional=self.num_rnn_direction == 2,
                                   batch_first=True)
                input_size += self.rnn_hidden_size
            elif sent_emb_method == 'cnn':
                self.out_channel = 50  # TODO: add param
                self.kernel_size = 3
                padding = (self.kernel_size - 1) // 2
                self.conv = nn.Conv1d(sent_emb_size, self.out_channel, self.kernel_size, stride=1, padding=padding)
                input_size += self.out_channel
            elif sent_emb_method == 'avg':
                input_size += sent_emb_size

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


    def combine_sent_emb(self,
                         lookup,
                         sent_ind,  # SHAPE: (batch_size, num_sent, num_words)
                         sent_count):  # SHAPE: (batch_size, num_sent)
        bs, ns, nw = sent_ind.size()
        # SHAPE: (batch_size * num_sent, num_words)
        sent_ind = sent_ind.view(-1, nw)
        # SHAPE: (batch_size * num_sent, num_words)
        token_mask = sent_ind.ne(self.padding_idx).float()
        # SHAPE: (batch_size * num_sent)
        sent_len = token_mask.sum(-1)
        sent_mask = sent_len.eq(0).long()
        # some sent is empty, pretend there is one token
        sent_len = torch.clamp(sent_len, min=1)

        # SHAPE: (batch_size * num_sent, num_words, emb_size)
        sent_emb = lookup(sent_ind)

        if self.sent_emb_method == 'rnn':
            packed_sent_emb = pack_padded_sequence(sent_emb, sent_len, batch_first=True, enforce_sorted=False)
            output, (last_h, last_c) = self.rnn(packed_sent_emb)
            #output, _ = pad_packed_sequence(output, batch_first=True)

            # SHAPE: (layer, dire, batch_size * num_sent, hidden_size)
            last_h = last_h.view(self.num_rnn_layer, self.num_rnn_direction, -1, self.rnn_hidden_size)
            # SHAPE: (batch_size, num_sent, hidden_size)
            last_h = last_h[-1][0].view(bs, ns, self.rnn_hidden_size)
            sent_emb = last_h * sent_mask.view(bs, ns, 1).float()  # mask out empty sent

        elif self.sent_emb_method == 'cnn':
            # SHAPE: (batch_size * num_sent, emb_size, num_words)
            sent_emb = sent_emb.permute(0, 2, 1)
            # SHAPE: (batch_size * num_sent, num_words, out_channel)
            sent_emb = self.conv(sent_emb).permute(0, 2, 1)
            sent_emb = sent_emb - (1 - token_mask).unsqueeze(-1) * 1e+10
            # SHAPE: (batch_size * num_sent, out_channel)
            sent_emb = sent_emb.max(1)[0].view(bs, ns, -1)

        elif self.sent_emb_method == 'avg':
            token_mask = token_mask.unsqueeze(-1)
            sent_emb = (sent_emb * token_mask).sum(1) / (token_mask.sum(1) + 1e-10)
            # SHAPE: (batch_size, num_sent, emb_size)
            sent_emb = sent_emb.view(bs, ns, -1)

        # sum across sentences
        sent_count = sent_count.float()
        sent_count = sent_count / (sent_count.sum(-1, keepdim=True) + 1e-10)
        if self.use_weight:
            sent_emb = (sent_emb * sent_count.float().unsqueeze(-1)).sum(1)
        else:
            sent_emb = sent_emb.sum(1)

        return sent_emb


    def forward(self, head, tail, labels,
                label_head=None, label_tail=None, tbow_ind=None, tbow_count=None, anc_labels=None,
                tbow_ind2=None, tbow_count2=None, sent_ind=None, sent_count=None):
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

        if sent_ind is not None and sent_count is not None:
            sent_emb = self.combine_sent_emb(self.sent_emb, sent_ind, sent_count)
            emb = torch.cat([emb, sent_emb], -1)

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


def get_sent_ind(batch, max_num=0, max_len=None):
    max_len_in_batch = np.max(np.array([[len(b[i][0]) if i < len(b) else 0 for i in range(max_num)] for b in batch]))
    # at least one token
    if max_len:
        max_len = max(min(max_len, max_len_in_batch), 1)
    else:
        max_len = max(max_len_in_batch, 1)
    word_ind_tensor = torch.tensor([[[b[i][0][j] if j < len(b[i][0]) else 0 for j in range(max_len)]
                                     if i < len(b) else ([0] * max_len) for i in range(max_num)] for b in batch])
    word_count_tensor = torch.tensor([[b[i][1] if i < len(b) else 0 for i in range(max_num)] for b in batch])
    return word_ind_tensor, word_count_tensor


def data2tensor(batch, emb_id2ind, only_prop=False, num_occs=10, device=None,
                label_samples=None, use_tbow=0, use_tbow2=0, use_sent=0,
                use_anc=False, num_occs_label=10, max_sent_len=128):  # TODO: add param
    if use_tbow and not use_tbow2:
        batch, tbow_batch = zip(*batch)
    elif use_tbow2:
        batch, tbow_batch, tbow_batch2 = zip(*batch)
    elif use_sent:
        batch, sent_batch = zip(*batch)

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
    tbow_ind2 = tbow_count2 = None
    sent_ind = sent_count = None,
    if use_tbow:
        tbow_ind, tbow_count = get_tbow_ind(tbow_batch, max_num=use_tbow)
        tbow_ind = tbow_ind.to(device)
        tbow_count = tbow_count.to(device)
    if use_tbow2:
        tbow_ind2, tbow_count2 = get_tbow_ind(tbow_batch2, max_num=use_tbow)
        tbow_ind2 = tbow_ind2.to(device)
        tbow_count2 = tbow_count2.to(device)
    if use_sent:
        sent_ind, sent_count = get_sent_ind(sent_batch, max_num=use_sent, max_len=max_sent_len)
        sent_ind = sent_ind.to(device)
        sent_count = sent_count.to(device)

    return head, tail, labels, label_head, label_tail, tbow_ind, tbow_count, anc_labels, \
           tbow_ind2, tbow_count2, sent_ind, sent_count


def samples2tensors(samples, batch_size, emb_id2ind,
                    num_occs, device, only_prop, label_samples=None,
                    use_tbow=0, use_tbow2=0, use_sent=0, use_anc=False, num_occs_label=0):
    tensors: List = []
    for batch in range(0, len(samples), batch_size):
        batch = samples[batch:batch + batch_size]
        tensor = data2tensor(
            batch, emb_id2ind, only_prop=only_prop, num_occs=num_occs, device=device,
            label_samples=label_samples, use_tbow=use_tbow, use_tbow2=use_tbow2, use_sent=use_sent,
            use_anc=use_anc, num_occs_label=num_occs_label)
        tensors.append(tensor)
    return tensors


def one_epoch(model, optimizer, samples, tensors, batch_size, emb_id2ind,
              num_occs, device, only_prop, is_train, label_samples=None,
              use_tbow=0, use_anc=False, num_occs_label=0):
    if is_train:
        model.train()
        shuffle(samples)
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
                  use_tbow=0, tbow_emb_size=50, word_emb_file=None, suffix='.tbow',  # tbow 1
                  use_tbow2=0, tbow_emb_size2=50, word_emb_file2=None, suffix2='.tbow',  # tbow 2
                  use_sent=0, sent_emb_size=50, sent_emb_file=None, sent_suffix='.sent',  # sent
                  only_tbow=False, renew_word_emb=False, output_pred=False, use_ancestor=False, filter_labels=False,
                  acc_topk=1, use_weight=False, only_one_sample_per_prop=False, optimizer='adam', use_gnn=None,
                  sent_emb_method='cnn'):
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

    train_samples_li, dev_samples_li, test_samples_li = [train_samples], [dev_samples], [test_samples]
    for split, sl in [('train', train_samples_li), ('dev', dev_samples_li), ('test', test_samples_li)]:
        for is_use, suff in [(use_tbow, suffix), (use_tbow2, suffix)]:
            if is_use:
                samples_more = read_tbow_file(os.path.join(data_dir, split + suff))
                assert len(samples_more) == len(sl[0])
                sl.append(samples_more)
        for is_use, suff in [(use_sent, sent_suffix)]:
            if is_use:
                samples_more = read_sent_file(os.path.join(data_dir, split + suff))
                assert len(samples_more) == len(sl[0])
                sl.append(samples_more)

    if len(train_samples_li) > 1:
        train_samples = list(zip(*train_samples_li))
        dev_samples = list(zip(*dev_samples_li))
        test_samples = list(zip(*test_samples_li))

    vocab_size, vocab_size2, sent_vocab_size = None, None, None
    word_emb, word_emb2, sent_emb = None, None, None
    if use_tbow:
        if word_emb_file:
            word_emb_id2ind, word_emb = read_embeddings_from_text_file(
                word_emb_file, debug=False, emb_size=tbow_emb_size, first_line=False, use_padding=True, split_char=' ')
            vocab_size = len(word_emb_id2ind)
            if renew_word_emb:
                word_emb = None
        else:
            vocab_size = len(load_tsv_as_dict(os.path.join(data_dir, '{}.vocab'.format(suffix.lstrip('.')))))

    if use_tbow2:
        if word_emb_file2:
            word_emb_id2ind2, word_emb2 = read_embeddings_from_text_file(
                word_emb_file2, debug=False, emb_size=tbow_emb_size2, first_line=False, use_padding=True, split_char=' ')
            vocab_size2 = len(word_emb_id2ind2)
            if renew_word_emb:
                word_emb2 = None
        else:
            vocab_size2 = len(load_tsv_as_dict(os.path.join(data_dir, '{}.vocab'.format(suffix2.lstrip('.')))))

    if use_sent:
        if sent_emb_file:
            sent_emb_id2ind, sent_emb = read_embeddings_from_text_file(
                sent_emb_file, debug=False, emb_size=sent_emb_size, first_line=False, use_padding=True, split_char=' ')
            sent_vocab_size = len(sent_emb_id2ind)
        else:
            sent_vocab_size = len(load_tsv_as_dict(os.path.join(data_dir, '{}.vocab'.format(sent_suffix.lstrip('.')))))

    print('vocab size 1 {}'.format(vocab_size))
    print('vocab size 2 {}'.format(vocab_size2))
    print('sent vocab size {}'.format(sent_vocab_size))

    if filter_leaves:
        print('filter leaves')
        filter_pids = set(leaves)
        if use_tbow or use_sent:
            train_samples = [s for s in train_samples if s[0][0][0] not in filter_pids]
            dev_samples = [s for s in dev_samples if s[0][0][0] not in filter_pids]
            test_samples = [s for s in test_samples if s[0][0][0] not in filter_pids]
        else:
            train_samples = [s for s in train_samples if s[0][0] not in filter_pids]
            dev_samples = [s for s in dev_samples if s[0][0] not in filter_pids]
            test_samples = [s for s in test_samples if s[0][0] not in filter_pids]

    if filter_labels:
        print('filter labels')
        if use_tbow or use_sent:
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

        if use_tbow or use_sent:
            key_func = lambda s: s[0][0][0]
        else:
            key_func = lambda s: s[0][0]
        train_samples = filter_first(train_samples, key_func=key_func)
        dev_samples = filter_first(dev_samples, key_func=key_func)
        test_samples = filter_first(test_samples, key_func=key_func)

    if only_test_on:
        if use_tbow or use_sent:
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

    if use_gnn:
        graph_data, meta_data = build_graph(
            train_samples, dev_samples, test_samples,
            'data/see_also.tsv', os.path.join(data_dir, 'subprops'),
            emb_id2ind, emb)
        graph_data = dict((k, v.to(device)) for k, v in graph_data.items())

        emb_model = EmbGnnModel(feat_size=input_size, hidden_size=hidden_size,
                                num_class=len(label2ind), dropout=dropout, method=use_gnn)
        emb_model.to(device)

        optimizer = torch.optim.RMSprop(emb_model.parameters(), lr=lr)

        last_metric, last_count = None, 0
        metrics = []
        for e in range(epoch):
            # train
            emb_model.train()
            train_logits, _, _, train_loss, _, _ = emb_model(**graph_data)
            train_logits = train_logits.detach().cpu().numpy()
            train_pred = [(pid, train_logits[i], l) for i, (pid, l) in
                          enumerate(zip(meta_data['train_pids'], meta_data['train_labels']))]

            if train_loss.requires_grad:
                optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(emb_model.parameters(), 1.0)
                optimizer.step()

            # eval
            emb_model.eval()
            _, dev_logits, test_logits, _, dev_loss, test_loss = emb_model(**graph_data)
            dev_logits = dev_logits.detach().cpu().numpy()
            test_logits = test_logits.detach().cpu().numpy()

            dev_pred = [(pid, dev_logits[i], l) for i, (pid, l) in
                        enumerate(zip(meta_data['dev_pids'], meta_data['dev_labels']))]
            test_pred = [(pid, test_logits[i], l) for i, (pid, l) in
                         enumerate(zip(meta_data['test_pids'], meta_data['test_labels']))]

            # metrics
            train_metric, train_ranks, train_pred_label, _ = accuracy_nway(
                train_pred, ind2label=ind2label, topk=acc_topk, num_classes=len(label2ind))
            dev_metric, dev_ranks, dev_pred_label, _ = accuracy_nway(
                dev_pred, ind2label=ind2label, topk=acc_topk, num_classes=len(label2ind))
            test_metric, test_ranks, test_pred_label, _ = accuracy_nway(
                test_pred, ind2label=ind2label, topk=acc_topk, num_classes=len(label2ind))

            print('train: {:>.3f}, {:>.3f} dev: {:>.3f}, {:>.3f} test: {:>.3f}, {:>.3f}'.format(
                train_loss.item(), train_metric,
                dev_loss.item(), dev_metric,
                test_loss.item(), test_metric))

            if early_stop and last_metric and last_metric > dev_metric:
                last_count += 1
                if last_count >= early_stop:
                    print('early stop')
                    break
            last_metric = dev_metric
            metrics.append(test_metric)

        return metrics, test_ranks, dev_ranks, train_ranks

    emb_model = EmbModel(emb, len(label2ind), len(anc2ind), input_size=input_size,
                         hidden_size=hidden_size, padding_idx=0,
                         dropout=dropout, only_prop=only_prop,
                         use_label=use_label,
                         vocab_size=vocab_size, tbow_emb_size=tbow_emb_size, word_emb=word_emb,
                         vocab_size2=vocab_size2, tbow_emb_size2=tbow_emb_size2, word_emb2=word_emb2,
                         sent_vocab_size=sent_vocab_size, sent_emb_size=sent_emb_size, sent_emb=sent_emb,
                         only_tbow=only_tbow, use_weight=use_weight, sent_emb_method=sent_emb_method)
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
        use_tbow=use_tbow, use_tbow2=use_tbow2, use_sent=use_sent, use_anc=use_ancestor, num_occs_label=num_occs_label)
    dev_tensors = samples2tensors(
        dev_samples, batch_size, emb_id2ind, num_occs, device, only_prop, label_samples=label_samples,
        use_tbow=use_tbow, use_tbow2=use_tbow2, use_sent=use_sent, use_anc=use_ancestor, num_occs_label=num_occs_label)
    test_tensors = samples2tensors(
        test_samples, batch_size, emb_id2ind, num_occs, device, only_prop, label_samples=label_samples,
        use_tbow=use_tbow, use_tbow2=use_tbow2, use_sent=use_sent, use_anc=use_ancestor, num_occs_label=num_occs_label)
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
