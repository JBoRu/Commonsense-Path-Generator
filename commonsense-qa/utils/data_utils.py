import pickle

# import dgl
import random
import string

import json
import numpy as np
import torch
from transformers import (GPT2Tokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, AlbertTokenizer)

from utils.tokenization_utils import *

GPT_SPECIAL_TOKENS = ["<PROMPT>","<START>","<PAD>","<END>","<SEP>"]

class BatchGenerator(object):
    def __init__(self, device, batch_size, indexes, qids, labels, tensors=[], lists=[], prompt_data=[]):
        self.device = device
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors = tensors
        self.lists = lists
        self.prompt_data = prompt_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes])
            batch_tensors = [self._to_device(x[batch_indexes]) for x in self.tensors]
            batch_lists = [self._to_device([x[i] for i in batch_indexes]) for x in self.lists]
            batch_prompt = [self._to_device(x[batch_indexes]) for x in self.prompt_data]
            batch_indexes = self._to_device(batch_indexes)
            yield tuple([batch_indexes, batch_qids, batch_labels, *batch_tensors, *batch_lists, batch_prompt])

    def _to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item) for item in obj]
        else:
            return obj.to(self.device)

class MixedBatchGenerator(object):
    def __init__(self, device, batch_size, csqa_indexes, nli_indexes, csqa_qids, nli_qids, csqa_labels, nli_labels,
                 csqa_tensors=[], nli_tensors=[], csqa_lists=[], nli_lists=[], csqa_prompt_data=[], nli_prompt_data=[]):
        self.device = device
        self.batch_size = batch_size
        self.csqa_indexes = csqa_indexes
        self.csqa_qids = csqa_qids
        self.csqa_labels = csqa_labels
        self.csqa_tensors = csqa_tensors
        self.csqa_lists = csqa_lists
        self.csqa_prompt_data = csqa_prompt_data
        self.nli_indexes = nli_indexes
        self.nli_qids = nli_qids
        self.nli_labels = nli_labels
        self.nli_tensors = nli_tensors
        self.nli_lists = nli_lists
        self.nli_prompt_data = nli_prompt_data

    def __len__(self):
        total_count = self.nli_indexes.size(0) + self.csqa_indexes.size(0)
        return (total_count - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes])
            batch_tensors = [self._to_device(x[batch_indexes]) for x in self.tensors]
            batch_lists = [self._to_device([x[i] for i in batch_indexes]) for x in self.lists]
            batch_prompt = [self._to_device(x[batch_indexes]) for x in self.prompt_data]
            batch_indexes = self._to_device(batch_indexes)
            yield tuple([batch_indexes, batch_qids, batch_labels, *batch_tensors, *batch_lists, batch_prompt])

    def _to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item) for item in obj]
        else:
            return obj.to(self.device)


class MultiGPUBatchGenerator(object):
    def __init__(self, device0, device1, batch_size, indexes, qids, labels, tensors0=[], lists0=[], tensors1=[], lists1=[]):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


class AdjDataBatchGenerator(object):
    def __init__(self, device, batch_size, indexes, qids, labels, tensors=[], lists=[], adj_empty=None, adj_data=None):
        self.device = device
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors = tensors
        self.lists = lists
        self.adj_empty = adj_empty
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        batch_adj = self.adj_empty  # (batch_size, num_choice, n_rel, n_node, n_node)
        batch_adj[:] = 0
        batch_adj[:, :, -1] = torch.eye(batch_adj.size(-1), dtype=torch.float32, device=self.device)
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes])
            batch_tensors = [self._to_device(x[batch_indexes]) for x in self.tensors]
            batch_lists = [self._to_device([x[i] for i in batch_indexes]) for x in self.lists]

            batch_adj[:, :, :-1] = 0
            for batch_id, global_id in enumerate(batch_indexes):
                for choice_id, (i, j, k) in enumerate(self.adj_data[global_id]):
                    batch_adj[batch_id, choice_id, i, j, k] = 1

            yield tuple([batch_qids, batch_labels, *batch_tensors, *batch_lists, batch_adj[:b - a]])

    def _to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item) for item in obj]
        else:
            return obj.to(self.device)


class MultiGPUAdjDataBatchGenerator(object):
    """
    this version DOES NOT add the identity matrix
    tensors0, lists0  are on device0
    tensors1, lists1, adj, labels  are on device1
    """

    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_empty=None, adj_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        self.adj_empty = adj_empty.to(self.device1)
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        batch_adj = self.adj_empty  # (batch_size, num_choice, n_rel, n_node, n_node)
        batch_adj[:] = 0
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            batch_adj[:] = 0
            for batch_id, global_id in enumerate(batch_indexes):
                for choice_id, (i, j, k) in enumerate(self.adj_data[global_id]):
                    batch_adj[batch_id, choice_id, i, j, k] = 1

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, batch_adj[:b - a]])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


class MultiGPUNxgDataBatchGenerator(object):
    """
    tensors0, lists0  are on device0
    tensors1, lists1, adj, labels  are on device1
    """

    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], graph_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        self.graph_data = graph_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            # qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            flat_graph_data = sum(self.graph_data, [])
            concept_mapping_dicts = []
            acc_start = 0
            for g in flat_graph_data:
                concept_mapping_dict = {}
                for index, cncpt_id in enumerate(g.ndata['cncpt_ids']):
                    concept_mapping_dict[int(cncpt_id)] = acc_start + index
                acc_start += len(g.nodes())
                concept_mapping_dicts.append(concept_mapping_dict)
            batched_graph = dgl.batch(flat_graph_data)
            batched_graph.ndata['cncpt_ids'] = batched_graph.ndata['cncpt_ids'].to(self.device1)

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_tensors1, *batch_lists0, *batch_lists1, batched_graph, concept_mapping_dicts])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


def load_2hop_relational_paths_old(input_jsonl_path, max_tuple_num, num_choice=None):
    with open(input_jsonl_path, 'r') as fin:
        rpath_data = [json.loads(line) for line in fin]
    n_samples = len(rpath_data)
    qa_data = torch.zeros((n_samples, max_tuple_num, 2), dtype=torch.long)
    rel_data = torch.zeros((n_samples, max_tuple_num), dtype=torch.long)
    num_tuples = torch.zeros((n_samples,), dtype=torch.long)
    for i, data in enumerate(tqdm(rpath_data, total=n_samples, desc='loading QA pairs')):
        cur_qa = []
        cur_rel = []
        for dic in data['paths']:
            if len(dic['rel']) == 1:
                cur_qa.append([dic['qc'], dic['ac']])
                cur_rel.append(dic['rel'][0])
            elif len(dic['rel']) == 2:
                cur_qa.append([dic['qc'], dic['ac']])
                cur_rel.append(34 + dic['rel'][0] * 34 + dic['rel'][1])
            else:
                raise ValueError('Invalid path length')
        assert len(cur_qa) == len(cur_rel)
        cur_qa, cur_rel = cur_qa[:min(max_tuple_num, len(cur_qa))], cur_rel[:min(max_tuple_num, len(cur_rel))]
        qa_data[i][:len(cur_qa)] = torch.tensor(cur_qa) if cur_qa else torch.zeros((0, 2), dtype=torch.long)
        rel_data[i][:len(cur_rel)] = torch.tensor(cur_rel) if cur_rel else torch.zeros((0,), dtype=torch.long)
        num_tuples[i] = (len(cur_qa) + len(cur_rel)) // 2  # code style suggested by kiwiser

    if num_choice is not None:
        qa_data = qa_data.view(-1, num_choice, max_tuple_num, 2)
        rel_data = rel_data.view(-1, num_choice, max_tuple_num)
        num_tuples = num_tuples.view(-1, num_choice)

    return qa_data, rel_data, num_tuples


def load_2hop_relational_paths(rpath_jsonl_path, cpt_jsonl_path=None, emb_pk_path=None,
                               max_tuple_num=200, num_choice=None, node_feature_type=None):
    with open(rpath_jsonl_path, 'r') as fin:
        rpath_data = [json.loads(line) for line in fin]

    with open(cpt_jsonl_path, 'rb') as fin:
        adj_data = pickle.load(fin)  # (adj, concepts, qm, am)

    n_samples = len(rpath_data)
    qa_data = torch.zeros((n_samples, max_tuple_num, 2), dtype=torch.long)
    rel_data = torch.zeros((n_samples, max_tuple_num), dtype=torch.long)
    num_tuples = torch.zeros((n_samples,), dtype=torch.long)

    all_masks = []
    for i, (data, adj) in enumerate(tqdm(zip(rpath_data, adj_data), total=n_samples, desc='loading QA pairs')):
        concept_ids = adj[1]
        ori_cpt2idx = {c: i for (i, c) in enumerate(concept_ids)}
        qa_mask = np.zeros(len(concept_ids), dtype=np.bool)

        cur_qa = []
        cur_rel = []
        for dic in data['paths']:
            if len(dic['rel']) == 1:
                cur_qa.append([dic['qc'], dic['ac']])
                cur_rel.append(dic['rel'][0])
            elif len(dic['rel']) == 2:
                cur_qa.append([dic['qc'], dic['ac']])
                cur_rel.append(34 + dic['rel'][0] * 34 + dic['rel'][1])
            else:
                raise ValueError('Invalid path length')
            qa_mask[ori_cpt2idx[dic['qc']]] = True
            qa_mask[ori_cpt2idx[dic['ac']]] = True
            if len(cur_qa) >= max_tuple_num:
                break
        assert len(cur_qa) == len(cur_rel)
        all_masks.append(qa_mask)

        if len(cur_qa) > 0:
            qa_data[i][:len(cur_qa)] = torch.tensor(cur_qa)
            rel_data[i][:len(cur_rel)] = torch.tensor(cur_rel)
            num_tuples[i] = (len(cur_qa) + len(cur_rel)) // 2  # code style suggested by kiwisher

    if emb_pk_path is not None:  # use contexualized node features
        with open(emb_pk_path, 'rb') as fin:
            all_embs = pickle.load(fin)
        assert len(all_embs) == len(all_masks) == n_samples
        max_cpt_num = max(mask.sum() for mask in all_masks)
        if node_feature_type in ('cls', 'mention'):
            emb_dim = all_embs[0].shape[1] // 2
        else:
            emb_dim = all_embs[0].shape[1]
        emb_data = torch.zeros((n_samples, max_cpt_num, emb_dim), dtype=torch.float)
        for idx, (mask, embs) in enumerate(zip(all_masks, all_embs)):
            assert not any(mask[embs.shape[0]:])
            masked_concept_ids = adj_data[idx][1][mask]
            masked_embs = embs[mask[:embs.shape[0]]]
            cpt2idx = {c: i for (i, c) in enumerate(masked_concept_ids)}
            for tuple_idx in range(num_tuples[idx].item()):
                qa_data[idx, tuple_idx, 0] = cpt2idx[qa_data[idx, tuple_idx, 0].item()]
                qa_data[idx, tuple_idx, 1] = cpt2idx[qa_data[idx, tuple_idx, 1].item()]
            if node_feature_type in ('cls',):
                masked_embs = masked_embs[:, :emb_dim]
            elif node_feature_type in ('mention',):
                masked_embs = masked_embs[:, emb_dim:]
            emb_data[idx, :masked_embs.shape[0]] = torch.tensor(masked_embs)
            assert (qa_data[idx, :num_tuples[idx]] < masked_embs.shape[0]).all()

    if num_choice is not None:
        qa_data = qa_data.view(-1, num_choice, max_tuple_num, 2)
        rel_data = rel_data.view(-1, num_choice, max_tuple_num)
        num_tuples = num_tuples.view(-1, num_choice)
        if emb_pk_path is not None:
            emb_data = emb_data.view(-1, num_choice, *emb_data.size()[1:])

    flat_rel_data = rel_data.view(-1, max_tuple_num)
    flat_num_tuples = num_tuples.view(-1)
    valid_mask = (torch.arange(max_tuple_num) < flat_num_tuples.unsqueeze(-1)).float()
    n_1hop_paths = ((flat_rel_data < 34).float() * valid_mask).sum(1)
    n_2hop_paths = ((flat_rel_data >= 34).float() * valid_mask).sum(1)
    print('| #paths: {} | average #1-hop paths: {} | average #2-hop paths: {} | #w/ 1-hop {} | #w/ 2-hop {} |'.format(flat_num_tuples.float().mean(0), n_1hop_paths.mean(), n_2hop_paths.mean(),
                                                                                                                      (n_1hop_paths > 0).float().mean(), (n_2hop_paths > 0).float().mean()))
    return (qa_data, rel_data, num_tuples, emb_data) if emb_pk_path is not None else (qa_data, rel_data, num_tuples)


def load_tokenized_statements(tokenized_path, num_choice, max_seq_len, freq_cutoff, vocab=None):
    with open(tokenized_path, 'r', encoding='utf-8') as fin:
        sents = [line.strip() for line in fin]

    if vocab is None:
        vocab = WordVocab(sents=sents, freq_cutoff=freq_cutoff)
        for tok in EXTRA_TOKS:
            vocab.add_word(tok)

    statement_data = torch.full((len(sents), max_seq_len), vocab.w2idx[PAD_TOK], dtype=torch.int64)
    statement_len = torch.full((len(sents),), 0, dtype=torch.int64)

    for i, sent in tqdm(enumerate(sents), total=len(sents), desc='loading tokenized'):
        word_ids = [vocab.w2idx[w] if w in vocab else vocab.w2idx[UNK_TOK] for w in (sent.split(' ')[:(max_seq_len - 1)] + [EOS_TOK])]
        if len(word_ids) > 0:
            statement_data[i][:len(word_ids)] = torch.tensor(word_ids)
            statement_len[i] = len(word_ids)

    statement_data = statement_data.view(-1, num_choice, max_seq_len)
    statement_len = statement_len.view(-1, num_choice)
    return statement_data, statement_len, vocab


def load_adj_data(adj_pk_path, max_node_num, num_choice, emb_pk_path=None):
    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)

    n_samples = len(adj_concept_pairs)
    adj_data = []
    adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
    concept_ids = torch.zeros((n_samples, max_node_num), dtype=torch.long)
    node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)

    if emb_pk_path is not None:
        with open(emb_pk_path, 'rb') as fin:
            all_embs = pickle.load(fin)
        emb_data = torch.zeros((n_samples, max_node_num, all_embs[0].shape[1]), dtype=torch.float)

    adj_lengths_ori = adj_lengths.clone()
    for idx, (adj, concepts, qm, am) in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
        num_concept = min(len(concepts), max_node_num)
        adj_lengths_ori[idx] = len(concepts)
        if emb_pk_path is not None:
            embs = all_embs[idx]
            assert embs.shape[0] >= num_concept
            emb_data[idx, :num_concept] = torch.tensor(embs[:num_concept])
            concepts = np.arange(num_concept)
        else:
            concepts = concepts[:num_concept]
        concept_ids[idx, :num_concept] = torch.tensor(concepts)  # note : concept zero padding is disabled

        adj_lengths[idx] = num_concept
        node_type_ids[idx, :num_concept][torch.tensor(qm, dtype=torch.uint8)[:num_concept]] = 0
        node_type_ids[idx, :num_concept][torch.tensor(am, dtype=torch.uint8)[:num_concept]] = 1
        ij = torch.tensor(adj.row, dtype=torch.int64)
        k = torch.tensor(adj.col, dtype=torch.int64)
        n_node = adj.shape[1]
        half_n_rel = adj.shape[0] // n_node
        i, j = ij // n_node, ij % n_node
        mask = (j < max_node_num) & (k < max_node_num)
        i, j, k = i[mask], j[mask], k[mask]
        i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
        adj_data.append((i, j, k))  # i, j, k are the coordinates of adj's non-zero entries

    print('| ori_adj_len: {:.2f} | adj_len: {:.2f} |'.format(adj_lengths_ori.float().mean().item(), adj_lengths.float().mean().item()) +
          ' prune_rate: {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
          ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                      (node_type_ids == 1).float().sum(1).mean().item()))

    concept_ids, node_type_ids, adj_lengths = [x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, adj_lengths)]
    if emb_pk_path is not None:
        emb_data = emb_data.view(-1, num_choice, *emb_data.size()[1:])
    adj_data = list(map(list, zip(*(iter(adj_data),) * num_choice)))

    if emb_pk_path is None:
        return concept_ids, node_type_ids, adj_lengths, adj_data, half_n_rel * 2 + 1
    return concept_ids, node_type_ids, adj_lengths, emb_data, adj_data, half_n_rel * 2 + 1


def construct_prompt_with_input(dataset):
    prompt_dataset = []
    for question, *choices, label in dataset:
        question = "Question: " + question
        choices = []
        for choic_idx, choice in enumerate(choices):
            choice = "Answer: " + choice
            # prompt = " ".join([prompt_token]*3)
            # sentence = prompt + " " + Q + " " + C + " " + prompt
            # tokens = tokenizer.tokenize(sentence)
            # choices_features.append(tokens)
            choices.append(choice)
        prompt_dataset.append((question, choices, label))
    return prompt_dataset


def load_gpt_input_tensors(model_name, pattern_type, statement_jsonl_path, max_seq_length, prompt_token_num):
    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            else:
                print("the sentence length more than max_len!")
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def load_qa_dataset(dataset_path):
        """ Output a list of tuples(sample_id, question, choices, label) """
        with open(dataset_path, "r", encoding="utf-8") as fin:
            output = []
            for line in fin:
                input_json = json.loads(line)
                label = ord(input_json.get("answerKey", "A")) - ord("A")
                output.append((input_json['id'], input_json["question"]["stem"], *[ending["text"] for ending in input_json["question"]["choices"]], label))
        return output

    def pre_process_datasets(dataset, num_choices, max_seq_length, special_tokens_ids_dict, prompt_token_num):
        """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

            To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
            input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        """
        prompt_token_ids = special_tokens_ids_dict['<PROMPT>']
        pad_token_ids = special_tokens_ids_dict['<PAD>']
        start_token_ids = special_tokens_ids_dict['<START>']
        end_token_ids = special_tokens_ids_dict['<END>']
        sep_token_ids = special_tokens_ids_dict['<SEP>']
        features = []
        n_batch = len(dataset)
        for i, sample in enumerate(dataset):
            choices_features = []
            q, mc_label = sample[0], sample[-1]
            choices = sample[1:-1]

            for j in range(len(choices)):
                c = choices[j]
                #### start q sep c end ####
                if pattern_type == -1:
                    if i == j == 0:
                        print("Using pattern of [START] q [SEP] c [END]")
                    special_tokens_count = 3
                    q = q
                    c = ' ' + c
                    tokens_q = tokenizer.tokenize(q)
                    tokens_c = tokenizer.tokenize(c)
                    _truncate_seq_pair(tokens_q, tokens_c, max_seq_length - special_tokens_count)
                    tokens_q_ids = tokenizer.convert_tokens_to_ids(tokens_q)
                    tokens_c_ids = tokenizer.convert_tokens_to_ids(tokens_c)
                    input_ids = [start_token_ids] + tokens_q_ids + [sep_token_ids] + tokens_c_ids + [end_token_ids]
                    qc_tokens_ids = input_ids
                #### _ _ _ q c _ _ _ #####
                elif pattern_type == 0:
                    if i == j == 0:
                        print("Using pattern of _ _ _ q c _ _ _")
                    q = "Question: " + q
                    c = " Answer: " + c
                    context = q + c
                    special_tokens_count = prompt_token_num
                    tokens_q = tokenizer.tokenize(q)
                    tokens_c = tokenizer.tokenize(c)
                    _truncate_seq_pair(tokens_q, tokens_c, max_seq_length - special_tokens_count)
                    qc_tokens = tokens_q + tokens_c
                    qc_tokens_ids = tokenizer.convert_tokens_to_ids(qc_tokens)
                    input_ids = [prompt_token_ids]*(int(prompt_token_num/2)) + qc_tokens_ids + [prompt_token_ids]*(int(prompt_token_num/2))
                #### _ _ _ q _ _ _ c _ _ _ ####
                elif pattern_type == 1:
                    if i == j == 0:
                        print("Using pattern of _ _ _ q _ _ _ c _ _ _")
                    q = "Question: " + q
                    c = "Answer: " + c
                    # context = q + c
                    special_tokens_count = prompt_token_num
                    tokens_q = tokenizer.tokenize(q)
                    tokens_c = tokenizer.tokenize(c)
                    _truncate_seq_pair(tokens_q, tokens_c, max_seq_length - special_tokens_count)
                    tokens_q = tokenizer.convert_tokens_to_ids(tokens_q)
                    tokens_c = tokenizer.convert_tokens_to_ids(tokens_c)
                    qc_tokens = tokens_q + tokens_c
                    qc_tokens_ids = tokenizer.convert_tokens_to_ids(qc_tokens)
                    input_ids = [prompt_token_ids]*(int(prompt_token_num/3)) + tokens_q + [prompt_token_ids]*(int(prompt_token_num/3)) + tokens_c + [prompt_token_ids]*(int(prompt_token_num/3))
                #### start _ _ _ q c _ _ _ answer ####
                elif pattern_type == 2:
                    if i == j == 0:
                        print("Using pattern of start _ _ _ q c _ _ _ answer")
                    q = "Question: " + q
                    c = " Answer: " + c
                    context = q + c
                    special_tokens_count = prompt_token_num
                    tokens_q = tokenizer.tokenize(q)
                    tokens_c = tokenizer.tokenize(c)
                    _truncate_seq_pair(tokens_q, tokens_c, max_seq_length - special_tokens_count)
                    qc_tokens = tokens_q + tokens_c
                    qc_tokens_ids = tokenizer.convert_tokens_to_ids(qc_tokens)
                    pos_label_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Yes"))
                    neg_label_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("No"))
                    mlm_label = 1 if j == int(mc_label) else 0
                    # start _ _ _ q c _ _ _ answer
                    if mlm_label:
                        input_ids = qc_tokens_ids[0:1] + [prompt_token_ids]*(int(prompt_token_num/2)) + qc_tokens_ids + \
                                    [prompt_token_ids]*(int(prompt_token_num/2)) + pos_label_ids
                    else:
                        input_ids = qc_tokens_ids[0:1] + [prompt_token_ids]*(int(prompt_token_num/2)) + qc_tokens_ids + \
                                    [prompt_token_ids] * (int(prompt_token_num/2)) + neg_label_ids
                #### Question: q Is it c ? _ _ _ ####
                elif pattern_type == 3:
                    if i == j == 0:
                        print("Using pattern of Question: q Is it c ? _ _ _")
                    q = "Question: " + q  # context is question
                    c = " Is it " + c + "?"  # ending is choice
                    special_tokens_count = prompt_token_num
                    tokens_q = tokenizer.tokenize(q)
                    tokens_c = tokenizer.tokenize(c)
                    _truncate_seq_pair(tokens_q, tokens_c, max_seq_length - special_tokens_count)
                    qc_tokens = tokens_q + tokens_c
                    qc_tokens_ids = tokenizer.convert_tokens_to_ids(qc_tokens)
                    input_ids = qc_tokens_ids + [prompt_token_ids] * (prompt_token_num)
                #### _ _ _ Question: q Is it c ? _ _ _ ####
                elif pattern_type == 4:
                    if i == j == 0:
                        print("Using pattern of _ _ _ Question: q Is it c ? _ _ _")
                    q = "Question: " + q  # context is question
                    c = " Is it " + c + "?"  # ending is choice
                    special_tokens_count = prompt_token_num
                    tokens_q = tokenizer.tokenize(q)
                    tokens_c = tokenizer.tokenize(c)
                    _truncate_seq_pair(tokens_q, tokens_c, max_seq_length - special_tokens_count)
                    qc_tokens = tokens_q + tokens_c
                    qc_tokens_ids = tokenizer.convert_tokens_to_ids(qc_tokens)
                    input_ids = [prompt_token_ids] * (int(prompt_token_num / 2)) + qc_tokens_ids + [
                        prompt_token_ids] * (int(prompt_token_num / 2))

                    # masked = " " + mask_token + ", it is!"
                    # sen_a = context + ending + masked
                    # if i == j == 0:
                    #     print("Using pattern of Question: q Is it c ? _ _ _ answer")
                    # context = "Question: " + context  # context is question
                    # ending = " Is it " + ending + "?"  # ending is choice
                    # masked = " " + mask_token + ", it is!"
                    # sen_a = context + ending + masked
                #### _ _ _ Question: q Is it c ? _ ####
                elif pattern_type == 5:
                    if i == j == 0:
                        print("Using pattern of _ _ _ Question: q Is it c ? _")
                    q = "Question: " + q  # context is question
                    c = " Is it " + c + "?"  # ending is choice
                    special_tokens_count = prompt_token_num
                    tokens_q = tokenizer.tokenize(q)
                    tokens_c = tokenizer.tokenize(c)
                    _truncate_seq_pair(tokens_q, tokens_c, max_seq_length - special_tokens_count)
                    qc_tokens = tokens_q + tokens_c
                    qc_tokens_ids = tokenizer.convert_tokens_to_ids(qc_tokens)
                    input_ids = [prompt_token_ids] * (prompt_token_num-1) + qc_tokens_ids + [prompt_token_ids]
                    # input_ids = [prompt_token_ids] * (prompt_token_num - 1) + qc_tokens_ids
                #### Question: q Is it c ? _ ####
                elif pattern_type == 6:
                    if i == j == 0:
                        print("Using pattern of Question: q Is it c ? end")
                    q = "Question: " + q  # context is question
                    c = " Is it " + c + "?"  # ending is choice
                    special_tokens_count = 1
                    tokens_q = tokenizer.tokenize(q)
                    tokens_c = tokenizer.tokenize(c)
                    _truncate_seq_pair(tokens_q, tokens_c, max_seq_length - special_tokens_count)
                    qc_tokens = tokens_q + tokens_c
                    qc_tokens_ids = tokenizer.convert_tokens_to_ids(qc_tokens)
                    input_ids = qc_tokens_ids + [end_token_ids]

                input_mask = [1]*len(input_ids)
                prompt_token_idx = [index for index, id in enumerate(input_ids) if id == prompt_token_ids]
                # assert len(prompt_token_idx) == special_tokens_count, f"%d %d"%(len(prompt_token_idx),special_tokens_count)
                block_flag = [0]*len(input_ids)
                for idx in prompt_token_idx:
                    block_flag[idx] = 1 # 1 for prompt placeholder
                mlm_mask = [0]*len(input_ids)
                if pattern_type == 2:
                    mlm_mask[-2] = 1
                else:
                    mlm_mask[-1] = 1

                pad_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([pad_token_ids]*pad_length)
                input_mask = input_mask + ([0]*pad_length)

                # Note: i not use these two variable, only for consistent with input format of Roberta
                output_mask = input_mask
                segment_ids = input_mask

                block_flag = block_flag + ([0] * pad_length)
                mlm_mask = mlm_mask + ([0] * pad_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = 1 if j == int(mc_label) else 0

                choices_features.append(
                    (qc_tokens_ids, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label))

            features.append(InputFeatures(example_id=i, choices_features=choices_features, label=mc_label))
        assert len(features) == n_batch
        return features

    def tokenize_and_encode(tokenizer, obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        else:
            return list(tokenize_and_encode(tokenizer, o) for o in obj)

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    if model_name == "gpt2-medium":
        path = "/mnt/nlp_model/gpt2-medium/"
    else:
        path = "/mnt/nlp_model/huggingface/gpt2/"

    print("Load tokenizer from %s" % (path))
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(GPT_SPECIAL_TOKENS)
    special_tokens_ids_dict = {}
    for token, id in zip(GPT_SPECIAL_TOKENS, special_tokens_ids):
        special_tokens_ids_dict[token] = id

    # [(id,q,cs,label),]
    dataset = load_qa_dataset(statement_jsonl_path)
    example_ids = [data[0] for data in dataset]
    dataset = [data[1:] for data in dataset]  # discard example ids
    num_choices = len(dataset[0]) - 2

    features = pre_process_datasets(dataset, num_choices, max_seq_length, special_tokens_ids_dict, prompt_token_num)
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)


def get_gpt_token_num():
    tokenizer = GPT2Tokenizer.from_pretrained('/mnt/nlp_model/huggingface/gpt2/')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    return len(tokenizer)


def load_bert_xlnet_roberta_input_tensors(args, statement_jsonl_path, model_type, model_name, max_seq_length):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                if "para" in json_dic:
                    contexts = json_dic["para"] + "[SEP]" + json_dic["question"]["stem"]
                else:
                    contexts = json_dic["question"]["stem"]
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
        return examples

    def convert_examples_to_features(pattern_type, examples, label_list, max_seq_length,
                                     tokenizer,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     sep_token_extra=False,
                                     mask_token='[MASK]',
                                     pad_token_segment_id=0,
                                     pad_on_left=False,
                                     pad_token=0,
                                     mask_padding_with_zero=True,
                                     num_prompt_token=0,
                                     prompt_token=["PROMPT"]):
        ''' Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet) '''
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            choices_features = []
            # roberta: <s>Question is A?</s></s>Answer is B.</s>
            # for one sample
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                # experiment with p-tuning format!
                if args.input_format in ['pg_kg_enc_as_prompt', 'GPT_kg_generator_as_prompt']:
                    kg_prefix = "According to: "
                    kg = [mask_token for i in range(num_prompt_token)]
                    kg = " ".join(kg)
                    context = ". Question: " + context # context is question
                    ending = " Is it " + ending + "?" # ending is choice
                    masked = " " + mask_token + ", it is!"
                    sen_a = kg_prefix + kg + context + ending + masked

                    tokens_a = tokenizer.tokenize(sen_a)
                    # print("Debug: sentence a is ", sen_a)
                    # print("Debug: tokenized sentence a is ", tokens_a)
                    tokens_b = []
                elif args.input_format == 'path-gen':
                    tokens_a = tokenizer.tokenize(context)
                    tokens_b = tokenizer.tokenize(example.question + " " + ending)
                elif args.input_format == 'manual_hard_prompt':
                    if ex_index == ending_idx == 0:
                        print("Using input pattern of 'Question: q Is it c? [mask], it is!' ")
                    context = "Question: " + context  # context is question
                    ending = " Is it " + ending + "?"  # ending is choice
                    masked = " " + mask_token + ", it is!"
                    sen_a = context + ending + masked

                    tokens_a = tokenizer.tokenize(sen_a)
                    # print("Debug: sentence a is ", sen_a)
                    # print("Debug: tokenized sentence a is ", tokens_a)
                    tokens_b = []
                elif args.input_format in ['soft_prompt_p_tuning', 'soft_prompt_p_tuning_classify']:
                    # no prompt gen
                    if pattern_type == -1:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'q <sep> c [mask]' ")
                        context = context  # context is question
                        ending = ending  # ending is choice
                        masked = mask_token
                        if sep_token_extra:
                            sen_a = context + " " + ending + ". " + sep_token + " " + sep_token + " " + masked
                        else:
                            sen_a = context + " " + ending + ". " + sep_token + " " + masked
                        sen_a = sen_a.lower()
                    # soft prompt gen
                    elif pattern_type == 0:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'Question: q Is it c? p1 p2 p3 p4 p5 p6 [mask]' ")
                        context = "Question: " + context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        soft_prompt = [mask_token for i in range(num_prompt_token)]
                        soft_prompt = " ".join(soft_prompt)
                        soft_prompt = soft_prompt
                        masked = mask_token + ", it is!"
                        sen_a = context + " " + ending + " " + soft_prompt + " " + masked
                    # soft prompt gen for 0
                    elif pattern_type == 10:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'q Is it c? sep p1 p2 p3 p4 p5 p6 [mask], it is!' ")
                        context = context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        soft_prompt = [prompt_token for i in range(num_prompt_token)]
                        soft_prompt = " ".join(soft_prompt)
                        masked = mask_token + ", it is!"
                        if sep_token_extra:
                            sen_a = context + " " + ending + " " + sep_token + " " + sep_token + " " + soft_prompt + " " + masked
                        else:
                            sen_a = context + " " + ending + " " + sep_token + " " + soft_prompt + " " + masked
                        sen_a = sen_a.lower()
                    # soft prompt gen
                    elif pattern_type == 1:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'p1 p2 p3 Question: q Answer: c p4 p5 p6 [mask]' ")
                        context = "Question: " + context  # context is question
                        ending = "Answer: " + ending  # ending is choice
                        soft_prompt = [mask_token for i in range(int(num_prompt_token/2))]
                        soft_prompt = " ".join(soft_prompt)
                        # soft_prompt = " " + soft_prompt
                        masked = " " + mask_token
                        sen_a = soft_prompt + " " + context + " " + ending + " " + soft_prompt + masked
                    # soft prompt gen
                    elif pattern_type == 2:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'p1 p2 p3 Question: q Is it c? p4 p5 p6 [mask]' ")
                        context = "Question: " + context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        soft_prompt = [mask_token for i in range(int(num_prompt_token / 2))]
                        soft_prompt = " ".join(soft_prompt)
                        masked = " " + mask_token + ", it is!"
                        sen_a = soft_prompt + " " + context + " " + ending + " " + soft_prompt + masked
                    # soft prompt gen
                    elif pattern_type == 3:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'p1 p2 p3 Question: q Is it c? [mask]' ")
                        context = "Question: " + context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        soft_prompt = [mask_token for i in range(num_prompt_token)]
                        soft_prompt = " ".join(soft_prompt)
                        masked = " " + mask_token + ", it is!"
                        sen_a = soft_prompt + " " + context + " " + ending + masked
                    # soft prompt cls
                    elif pattern_type == 4:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'Question: q Is it c? p1 p2 p3 p4 p5 p6' ")
                        context = "Question: " + context  # context is question
                        ending = " Is it " + ending + "?"  # ending is choice
                        soft_prompt = [mask_token for i in range(num_prompt_token)]
                        soft_prompt = " ".join(soft_prompt)
                        soft_prompt = " " + soft_prompt
                        sen_a = context + ending + soft_prompt
                    # soft prompt cls for 4
                    elif pattern_type == 11:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'q <sep> candidate answer is c _ _ _ _ _ _ mask' ")
                        context = context  # context is question
                        ending = "Candidate Answer is " + ending + "."  # ending is choice
                        soft_prompt = [mask_token for i in range(num_prompt_token)]
                        soft_prompt = " ".join(soft_prompt)
                        if sep_token_extra:
                            sen_a = context + " " + sep_token + " " + sep_token + " " + ending + " " + soft_prompt + mask_token
                        else:
                            sen_a = context + " " + sep_token + " " + ending + " " + soft_prompt + mask_token
                        sen_a = sen_a.lower()
                    # no prompt cls
                    elif pattern_type == 5:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'cls q sep c end' ")
                        context = context  # context is question
                        ending = ending # ending is choice
                        if sep_token_extra:
                            sen_a = context + " " + sep_token + " " + sep_token + " " + ending + "."
                        else:
                            sen_a = context + " " + sep_token + " " + ending + "."
                        sen_a = sen_a.lower()
                    # hard prompt gen
                    elif pattern_type == 6:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'Question: q Is it c? [mask], it is' ")
                        context = "Question: " + context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        masked = " " + mask_token + ", it is!"
                        sen_a = context + " " + ending + masked
                    # hard prompt gen from 6
                    elif pattern_type == 8:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'q <sep> Is it c? [mask], it is' ")
                        context = context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        masked = mask_token + ", it is!"
                        if sep_token_extra:
                            sen_a = context + " " + sep_token + " " + sep_token + " " + ending  + " " + masked
                        else:
                            sen_a = context + " " + sep_token + " " + ending + " " + masked
                        sen_a = sen_a.lower()
                    # hard prompt cls
                    elif pattern_type == 7:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'Question: q <sep> Anser: c' ")
                        context = "Question: " + context  # context is question
                        ending = "Answer: " + ending  # ending is choice
                        sen_a = context + " " + ending
                    # hard prompt cls from 7
                    elif pattern_type == 9:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'q <sep> candidate answer is c' ")
                        context = context  # context is question
                        ending = "Candidate Answer is " + ending + "."  # ending is choice
                        if sep_token_extra:
                            sen_a = context + " " + sep_token + " " + sep_token + " " + ending
                        else:
                            sen_a = context + " " + sep_token + " " + ending
                        sen_a = sen_a.lower()

                    tokens_a = tokenizer.tokenize(sen_a)
                    tokens_b = []

                # for roberta, it will be format of <s> X </s> or <s> A </s></s> B </s>
                # special_tokens_count = 4 if (sep_token_extra and bool(tokens_b)) else 3
                # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

                tokens = tokens_a + [sep_token]
                # if sep_token_extra and bool(tokens_b):
                #     # roberta uses an extra separator b/w pairs of sentences
                #     tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)

                if tokens_b:
                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids
                # till now, tokens become <cls> context </s>

                # convert to ids
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([cls_token, sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                # assert len(mask_token_index) == (num_prompt_token+1), "More than specified masked position in one example"
                block_flag = [0]*len(input_ids)
                mlm_mask = [0]*len(input_ids)
                if args.input_format in ['pg_kg_enc_as_prompt', 'soft_prompt_p_tuning', 'soft_prompt_p_tuning_classify', 'GPT_kg_generator_as_prompt']:
                    if pattern_type == 4: # no mask token and use cls at the head position
                        for idx in mask_token_index:
                            block_flag[idx] = 1  # 1 for prompt placeholder
                        mlm_mask[0] = 1
                    elif pattern_type == 5: # no prompt for class
                        mlm_mask[0] = 1
                    elif pattern_type in [7,9]: # hard prompt fot  class
                        mlm_mask[0] = 1
                    elif pattern_type == -1: # no prompt for gen
                        mlm_mask[mask_token_index[0]] = 1
                    elif pattern_type in [6, 8]: # hard prompt for gen
                        mlm_mask[mask_token_index[0]] = 1
                    else: # soft ptompt gen
                        for idx in mask_token_index[0:-1]:
                            block_flag[idx] = 1  # 1 for prompt placeholder
                        mlm_mask[mask_token_index[-1]] = 1 # 1 for masked token
                elif args.input_format == 'manual_hard_prompt':
                    if len(mask_token_index) == 1:
                        mlm_mask[mask_token_index[-1]] = 1  # 1 for masked token
                    else:
                        print("There are more than 1 mask in manual_hard_prompt format!")


                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    output_mask = ([1] * padding_length) + output_mask

                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    output_mask = output_mask + ([1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
                    block_flag = block_flag + ([0] * padding_length)
                    mlm_mask = mlm_mask + ([0] * padding_length)


                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = 1 if ending_idx == label else 0
                choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label))
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer}.get(model_type)
    if model_type == 'albert':
        path = '/mnt/nlp_model/albert-xxlarge-v2/'
    elif model_type == 'roberta':
        path = '/mnt/nlp_model/huggingface/roberta-large/'
    tokenizer = tokenizer_class.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_special_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(args.pattern_type, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer,
                                            cls_token_at_end=bool(model_type in ['xlnet']),  # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta']),
                                            mask_token=tokenizer.mask_token,
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                            sequence_b_segment_id=0 if model_type in ['roberta', 'albert'] else 1,
                                            num_prompt_token=args.prompt_token_num,
                                            prompt_token=prompt_token)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_roberta_input_tensors(args, statement_jsonl_path, model_type, model_name, max_seq_length, num_prompt_token):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                contexts = json_dic["question"]["stem"]
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
        return examples

    def convert_examples_to_features(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        ''' Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet) '''
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        mask_token = tokenizer.mask_token
        sep_token = tokenizer.sep_token
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            choices_features = []
            # roberta: <s>Question is A?</s></s>Answer is B.</s>
            # for one sample
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                # experiment with p-tuning format!
                if args.input_format in ['pg_kg_enc_as_prompt', 'GPT_kg_generator_as_prompt']:
                    kg_prefix = "According to: "
                    kg = [mask_token for i in range(num_prompt_token)]
                    kg = " ".join(kg)
                    context = ". Question: " + context # context is question
                    ending = " Is it " + ending + "?" # ending is choice
                    masked = " " + mask_token + ", it is!"
                    sen_a = kg_prefix + kg + context + ending + masked

                    tokens_a = tokenizer.tokenize(sen_a)
                    # print("Debug: sentence a is ", sen_a)
                    # print("Debug: tokenized sentence a is ", tokens_a)
                    tokens_b = []
                elif args.input_format == 'path-gen':
                    tokens_a = tokenizer.tokenize(context)
                    tokens_b = tokenizer.tokenize(example.question + " " + ending)
                elif args.input_format == 'manual_hard_prompt':
                    if ex_index == ending_idx == 0:
                        print("Using input pattern of 'Question: q Is it c? [mask], it is!' ")
                    context = "Question: " + context  # context is question
                    ending = " Is it " + ending + "?"  # ending is choice
                    masked = " " + mask_token + ", it is!"
                    sen_a = context + ending + masked

                    tokens_a = tokenizer.tokenize(sen_a)
                    # print("Debug: sentence a is ", sen_a)
                    # print("Debug: tokenized sentence a is ", tokens_a)
                    tokens_b = []
                elif args.input_format in ['soft_prompt_p_tuning', 'soft_prompt_p_tuning_classify']:
                    # no prompt gen
                    if pattern_type == -1:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of '<s>question</s></s>answer</s></s>[mask]</s>' ")
                        context = context  # context is question
                        ending = ending  # ending is choice
                        masked = tokenizer.mask_token
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + ending + tokenizer.sep_token + tokenizer.sep_token + masked + tokenizer.sep_token
                    # no prompt cls
                    elif pattern_type == 5:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of '<s>question</s></s>answer</s>' ")
                        context = context  # context is question
                        ending = ending  # ending is choice
                        masked = tokenizer.mask_token
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + ending + tokenizer.sep_token
                    # hard prompt gen
                    elif pattern_type == 6:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'Question: q Is it c? [mask], it is' ")
                        context = "Question: " + context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        masked = " " + mask_token + ", it is!"
                        sen_a = context + " " + ending + masked
                    # hard prompt gen from 6
                    elif pattern_type == 8:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'q <sep> Is it c? [mask], it is' ")
                        context = context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        masked = mask_token + ", it is!"
                        sen_a = context + " " + sep_token + " " + sep_token + " " + ending + " " + masked
                    # hard prompt cls
                    elif pattern_type == 7:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'Question: q <sep> Anser: c' ")
                        context = "Question: " + context  # context is question
                        ending = "Answer: " + ending  # ending is choice
                        sen_a = context + " " + ending
                    # hard prompt cls from 7
                    elif pattern_type == 9:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'q <sep> candidate answer is c' ")
                        context = context  # context is question
                        ending = "Candidate Answer is " + ending + "."  # ending is choice
                        sen_a = context + " " + sep_token + " " + sep_token + " " + ending
                    # soft prompt gen
                    elif pattern_type == 0:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'Question: q Is it c? p1 p2 p3 p4 p5 p6 [mask]' ")
                        context = "Question: " + context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        soft_prompt = [mask_token for i in range(num_prompt_token)]
                        soft_prompt = " ".join(soft_prompt)
                        soft_prompt = soft_prompt
                        masked = mask_token + ", it is!"
                        sen_a = context + " " + ending + " " + soft_prompt + " " + masked
                    # soft prompt gen for 0
                    elif pattern_type == 10:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'q Is it c? sep p1 p2 p3 p4 p5 p6 [mask], it is!' ")
                        context = context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        soft_prompt = [prompt_token for i in range(num_prompt_token)]
                        soft_prompt = " ".join(soft_prompt)
                        masked = mask_token + ", it is!"
                        sen_a = context + " " + ending + " " + sep_token + " " + sep_token + " " + soft_prompt + " " + masked
                    # soft prompt gen
                    elif pattern_type == 1:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'p1 p2 p3 Question: q Answer: c p4 p5 p6 [mask]' ")
                        context = "Question: " + context  # context is question
                        ending = "Answer: " + ending  # ending is choice
                        soft_prompt = [mask_token for i in range(int(num_prompt_token/2))]
                        soft_prompt = " ".join(soft_prompt)
                        # soft_prompt = " " + soft_prompt
                        masked = " " + mask_token
                        sen_a = soft_prompt + " " + context + " " + ending + " " + soft_prompt + masked
                    # soft prompt gen
                    elif pattern_type == 2:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'p1 p2 p3 Question: q Is it c? p4 p5 p6 [mask]' ")
                        context = "Question: " + context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        soft_prompt = [mask_token for i in range(int(num_prompt_token / 2))]
                        soft_prompt = " ".join(soft_prompt)
                        masked = " " + mask_token + ", it is!"
                        sen_a = soft_prompt + " " + context + " " + ending + " " + soft_prompt + masked
                    # soft prompt gen
                    elif pattern_type == 3:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of 'p1 p2 p3 Question: q Is it c? [mask]' ")
                        context = "Question: " + context  # context is question
                        ending = "Is it " + ending + "?"  # ending is choice
                        soft_prompt = [mask_token for i in range(num_prompt_token)]
                        soft_prompt = " ".join(soft_prompt)
                        masked = " " + mask_token + ", it is!"
                        sen_a = soft_prompt + " " + context + " " + ending + masked
                    # soft prompt gen
                    elif pattern_type == 11:
                        if ex_index == ending_idx == 0:
                            print(
                                "Using input pattern of '<s>question</s></s>Candidate answer is c. _ _ _ _ _ _ [mask], it is.</s>' ")
                        context = context  # context is question
                        ending = "Candidate Answer is " + ending + "."  # ending is choice
                        soft_prompt = [prompt_token for i in range(num_prompt_token)]
                        soft_prompt = " ".join(soft_prompt)
                        masked = tokenizer.mask_token + ", it is."
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + ending + " " + soft_prompt + " " + masked + tokenizer.sep_token
                    # soft prompt cls
                    elif pattern_type == 4:
                        if ex_index == ending_idx == 0:
                            print("Using input pattern of '<s>_ _ _ _ _ _ question</s></s>Candidate answer is c.</s>' ")
                        context = context  # context is question
                        ending = "Candidate Answer is " + ending + "."
                        soft_prompt = [prompt_token for i in range(num_prompt_token)]
                        soft_prompt = " ".join(soft_prompt)
                        input = tokenizer.cls_token + soft_prompt + " " + context + tokenizer.sep_token + tokenizer.sep_token + ending + tokenizer.sep_token

                    input_tokens = tokenizer.tokenize(input)

                # convert to ids
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                segment_ids = [0] * len(input_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                block_flag = [0]*len(input_ids)
                mlm_mask = [0]*len(input_ids)
                if args.input_format in ['pg_kg_enc_as_prompt', 'soft_prompt_p_tuning', 'soft_prompt_p_tuning_classify', 'GPT_kg_generator_as_prompt']:
                    if pattern_type == 4: # no mask token and use cls at the head position
                        for idx in prompt_token_index:
                            block_flag[idx] = 1  # 1 for prompt placeholder
                        mlm_mask[0] = 1
                    elif pattern_type == 5: # no prompt for class
                        mlm_mask[0] = 1
                    elif pattern_type in [7,9]: # hard prompt fot  class
                        mlm_mask[0] = 1
                    elif pattern_type == -1: # no prompt for gen
                        mlm_mask[mask_token_index[0]] = 1
                    elif pattern_type in [6, 8]: # hard prompt for gen
                        mlm_mask[mask_token_index[0]] = 1
                    else: # soft ptompt gen
                        for idx in prompt_token_index:
                            block_flag[idx] = 1  # 1 for prompt placeholder
                        mlm_mask[mask_token_index[0]] = 1 # 1 for masked token
                elif args.input_format == 'manual_hard_prompt':
                    if len(mask_token_index) == 1:
                        mlm_mask[mask_token_index[-1]] = 1  # 1 for masked token
                    else:
                        print("There are more than 1 mask in manual_hard_prompt format!")


                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                block_flag = block_flag + ([0] * padding_length)
                mlm_mask = mlm_mask + ([0] * padding_length)


                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = 1 if ending_idx == label else 0
                choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label))
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/huggingface/roberta-large/'
    tokenizer = RobertaTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(args.pattern_type, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_roberta_input_tensors_for_nli(args, statement_jsonl_path, model_class, model_name, max_seq_length,
                                       num_prompt_token, num_label, cache=None):
    class InputExample(object):
        def __init__(self, example_id, precises, hypothesises, label=None):
            self.example_id = example_id
            self.precises = precises
            self.hypothesises = hypothesises
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file, num_label):
        # with open(input_file, "r") as f:
        #     examples = []
        #     for line in f.readlines():
        #         line = line.strip("\n").split("\t")
        #         id, pre, hyp, label = [s.strip() for s in line]
        #         premises = []
        #         hypothesises = []
        #         relations = []
        #         premises.append(pre)
        #         hypothesises.append(hyp)
        #         premises = premises * num_label
        #         hypothesises = hypothesises * num_label
        #         relations = ["contradiction", "neutral", "entailment"]
        #         examples.append(
        #             InputExample(
        #                 example_id=id,
        #                 precises=premises,
        #                 hypothesises=hypothesises,
        #                 relations=relations,
        #                 label=label
        #             ))
        # with open(input_file, "r") as f:
        #     examples = []
        #     for id, line in enumerate(f.readlines()):
        #         line = line.strip("\n")
        #         line_dict = json.loads(line)
        #         precise = line_dict["p"]
        #         c = line_dict["c"]
        #         n = line_dict["n"]
        #         e = line_dict["e"]
        #         shuffle_list = [{"c":c},{"n":n},{"e":e}]
        #         random.shuffle(shuffle_list)
        #         premises = []
        #         hypothesises = []
        #         label = 0
        #         for idx, hyp_dict in enumerate(shuffle_list):
        #             rel, hyp = [kv for kv in hyp_dict.items()][0]
        #             premises.append(precise)
        #             hypothesises.append(hyp)
        #             if rel=="e":
        #                 label = idx
        #         examples.append(
        #             InputExample(
        #                 example_id=id,
        #                 precises=premises,
        #                 hypothesises=hypothesises,
        #                 label=label
        #             )
        #         )
        with open(input_file, "r") as f:
            examples = []
            for id, line in enumerate(f.readlines()):
                line = line.strip("\n")
                line_dict = json.loads(line)
                precise = line_dict["p"]
                n = line_dict["n"]
                e = line_dict["e"]
                premises= []
                hypothesises = []
                premises.append(precise)
                hypothesises.append(e)
                for idx, hyp in enumerate(n):
                    premises.append(precise)
                    hypothesises.append(hyp)
                label = 0
                examples.append(
                    InputExample(
                        example_id=id,
                        precises=premises,
                        hypothesises=hypothesises,
                        label=label
                    )
                )
        return examples

    def convert_examples_to_features(pattern_type, examples, max_seq_length, tokenizer):
        label_map = {"contradiction":0, "neutral":1, "entailment":2}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        max_seq_length_real = 0
        for ex_index, example in enumerate(examples):
            label = example.label
            choices_features = []
            # roberta: <s>premise</s></s>statement</s>
            # for one sample
            for ending_idx, (precise, hypothesis) in enumerate(zip(example.precises, example.hypothesises)):
                if pattern_class == "hard-prompt-cls":
                    if pattern_idx == "0":
                        p = precise + "." if precise[-1].isalpha() else precise
                        h = hypothesis + "." if hypothesis[-1].isalpha() else hypothesis
                        input = tokenizer.cls_token + p + tokenizer.sep_token + tokenizer.sep_token + h + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        p = precise + "." if precise[-1].isalpha() else precise
                        h = hypothesis + "." if hypothesis[-1].isalpha() else hypothesis
                        h = "According to the previous sentence. " + "The answer is: " + h.lower()
                        input = tokenizer.cls_token + p + tokenizer.sep_token + tokenizer.sep_token + h + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "2":
                        p = precise + "." if precise[-1].isalpha() else precise
                        h = hypothesis + "." if hypothesis[-1].isalpha() else hypothesis
                        h = "According to the previous sentence. " + "The answer is: " + h.lower()
                        input = tokenizer.cls_token + p + tokenizer.sep_token + tokenizer.sep_token + h + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)

                if ex_index == ending_idx == 0:
                    print("Tokenized tokens is:")
                    print(input_tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                    ori = tokenizer.decode(input_ids)
                    print("Using input pattern of:")
                    print(ori)

                # convert to ids
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                max_seq_length_real = max(max_seq_length_real, len(input_ids))
                # not use segment_ids, only keep format consistent.
                segment_ids = [0] * len(input_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                prompt_token_index = []

                block_flag = [0]*len(input_ids)
                mlm_mask = [0]*len(input_ids)

                if pattern_class == "hard-prompt-cls":
                    mlm_mask[0] = 1

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                block_flag = block_flag + ([0] * padding_length)
                mlm_mask = mlm_mask + ([0] * padding_length)


                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                # if relation == label:
                #     mlm_label = 1
                #     label = ending_idx
                # else:
                #     mlm_label = 0
                mlm_label = 1 if ending_idx == label else 0
                choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label))
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        print("Max len is: %d , the setting is: %d ."%(max_seq_length_real, max_seq_length))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/huggingface/roberta-large/'
    tokenizer = RobertaTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    # if cache is not None and os.path.exists(cache):
    #     data_frame = torch.load(cache)
    #     example_ids, all_label, *data_tensors, prompt_data_tensors = data_frame
    # else:
    #     examples = read_examples(statement_jsonl_path, num_label)
    #     features = convert_examples_to_features(args.pattern_format, examples, max_seq_length, tokenizer)
    #     example_ids = [f.example_id for f in features]
    #     *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    #     assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    #     data_frame = [example_ids, all_label, *data_tensors, prompt_data_tensors]
    #     torch.save(data_frame, cache)
    examples = read_examples(statement_jsonl_path, num_label)
    features = convert_examples_to_features(args.pattern_format, examples, max_seq_length, tokenizer)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_albert_input_tensors_for_nli(args, statement_jsonl_path, model_class, model_name, max_seq_length,
                                       num_prompt_token, num_label, cache=None):
    class InputExample(object):
        def __init__(self, example_id, precises, hypothesises, label=None):
            self.example_id = example_id
            self.precises = precises
            self.hypothesises = hypothesises
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in
                choices_features
            ]
            self.label = label

    def read_examples(input_file, num_label):
        with open(input_file, "r") as f:
            examples = []
            for id, line in enumerate(f.readlines()):
                line = line.strip("\n")
                line_dict = json.loads(line)
                precise = line_dict["p"]
                n = line_dict["n"]
                e = line_dict["e"]
                premises = []
                hypothesises = []
                premises.append(precise)
                hypothesises.append(e)
                for idx, hyp in enumerate(n):
                    premises.append(precise)
                    hypothesises.append(hyp)
                label = 0
                examples.append(
                    InputExample(
                        example_id=id,
                        precises=premises,
                        hypothesises=hypothesises,
                        label=label
                    )
                )
        return examples

    def convert_examples_to_features(pattern_type, examples, max_seq_length, tokenizer):
        label_map = {"contradiction": 0, "neutral": 1, "entailment": 2}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        max_seq_length_real = 0
        for ex_index, example in enumerate(examples):
            label = example.label
            choices_features = []
            # alberta: [CLS] Question is A? [SEP] Answer is B. [SEP]
            # for one sample
            for ending_idx, (precise, hypothesis) in enumerate(zip(example.precises, example.hypothesises)):
                if pattern_class == "hard-prompt-cls":
                    if pattern_idx == "2":
                        # <s>question</s></s>According to: triple</s></s>Candidate answer is choice.</s>
                        sent1 = precise
                        sent2 = hypothesis
                        input = tokenizer.cls_token + " " + sent1 + " " + tokenizer.sep_token +\
                                " " + sent2 + " " + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)

                if ex_index == ending_idx == 0:
                    print("Tokenized tokens is:")
                    print(input_tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                    ori = tokenizer.decode(input_ids)
                    print("Using input pattern of %s "%(ori))

                # convert to ids
                # input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                sent1_tokens = tokenizer.tokenize(sent1)
                sent2_tokens = tokenizer.tokenize(sent2)
                sent1_ids = tokenizer.convert_tokens_to_ids(sent1_tokens)
                sent2_ids = tokenizer.convert_tokens_to_ids(sent2_tokens)
                input_ids = tokenizer.build_inputs_with_special_tokens(sent1_ids, sent2_ids)
                # not use segment_ids, only keep format consistent.
                # segment_ids = [0] * len(input_ids)
                segment_ids = tokenizer.create_token_type_ids_from_sequences(sent1_ids, sent2_ids)
                # assert len(input_ids) == len(segment_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                block_flag = [0]*len(input_ids)
                mlm_mask = [0]*len(input_ids)
                if pattern_class == "hard-prompt-cls":
                    mlm_mask[0] = 1

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                block_flag = block_flag + ([0] * padding_length)
                mlm_mask = mlm_mask + ([0] * padding_length)


                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = 1 if ending_idx == label else 0
                choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label))
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/albert-xxlarge-v2/'
    tokenizer = AlbertTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path, num_label)
    features = convert_examples_to_features(args.pattern_format, examples, max_seq_length, tokenizer)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_roberta_input_tensors_for_kcr(args, statement_jsonl_path, model_type, model_name, max_seq_length, num_prompt_token):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, triples, surfaces, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.triples = triples
            self.surfaces = surfaces
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            all = json.load(f)
            examples = []
            for json_dic in all:
                choices = json_dic['question']['choices']
                question_concept = json_dic['question']['question_concept']
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                context = json_dic["question"]["stem"]

                triples = []
                endings = []
                contexts = []
                surfaces = []
                for i in range(len(choices)):
                    choice = choices[i]
                    if choice['triple']:
                        triple_str = ' '.join(choice['triple'][0])
                        triples_temp = triple_str
                        surface = choice['surface']
                        surface_str = surface.replace('[','').replace(']','')
                    else:
                        triples_temp = question_concept + tokenizer.sep_token + tokenizer.sep_token + choice['text']
                        surface_str = ""
                    triples.append(triples_temp)
                    endings.append(choice['text'])
                    contexts.append(context)
                    surfaces.append(surface_str)
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=contexts,
                        question="",
                        endings=endings,
                        triples=triples,
                        surfaces=surfaces,
                        label=label
                    ))
        return examples

    def convert_examples_to_features(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        ''' Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet) '''
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            choices_features = []

            # hard prompt generation with all candidates
            cdx_list = ["A", "B", "C", "D", "E"]
            if pattern_class == "hard-prompt-gen" and pattern_idx == "1":
                # <s>question</s></s>c1 t1</s></s>c2 t2</s></s>c3 t3</s></s>Is the right answer c1. [MASK]</s>
                input = tokenizer.cls_token
                sep_token = tokenizer.sep_token
                for ending_idx, (context, ending, triple, cdx) in enumerate(
                        zip(example.contexts, example.endings, example.triples, cdx_list)):
                    choice_feature = ending + ": " + triple
                    if ending_idx == 0:
                        input = input + context + sep_token + sep_token + choice_feature
                    else:
                        input = input + sep_token + sep_token + choice_feature
                prefix = input + sep_token + sep_token

            # roberta: <s>Question is A?</s></s>Answer is B.</s>
            # for one sample
            for ending_idx, (context, ending, triple, surface) in \
                    enumerate(zip(example.contexts, example.endings, example.triples, example.surfaces)):
                # no prompt cls
                if pattern_class == "no-prompt-cls":
                    if pattern_idx == "0":
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + triple + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                elif pattern_class == "hard-prompt-cls":
                    if pattern_idx == "0":
                        # <s>question</s></s>Candidate answer is choice.</s></s>triple</s>
                        context = context
                        triple = triple
                        choice = "Candidate answer is " + ending + "."
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                choice + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        # <s>question</s></s>According to: triple</s></s>Candidate answer is choice.</s>
                        context = context
                        triple = "According to: " + triple + "."
                        choice = "Candidate answer is " + ending + "."
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + \
                                choice + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "2":
                        # <s>question</s></s>According to suface, the answer is choice</s></s>triple</s>
                        context = context
                        surface = surface
                        if surface!='':
                            choice = "According to " + surface.lower() + ". " + "The answer is: " + ending.lower() + "."
                        else:
                            choice = "The answer is: " + ending.lower() + "."
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                choice + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "3":
                        # <s>question</s></s>The answer is choice, because</s></s>triple</s>
                        context = context
                        surface = surface
                        if surface!='':
                            choice = "The answer is: " + ending.lower() + ", " + "because " + surface.lower() + "."
                        else:
                            choice = "The answer is: " + ending.lower() + "."
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                choice + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                elif pattern_class == "hard-prompt-gen":
                    if pattern_idx == "0":
                        # <s>question</s></s>According to triple</s></s>Is choice the answer? [MASK], it is.
                        context = context
                        triple = "According to " + triple
                        choice = "Is " + ending + " the answer?"
                        mask = tokenizer.mask_token + ", it is."
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + choice + \
                                " " + mask + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        # <s>question</s></s>c1 t1</s></s>c2 t2</s></s>c3 t3</s></s>According to the choices, is the right answer c1. [MASK]</s>
                        context = prefix
                        choice = "According to the choices, is " + ending + " the right answer?"
                        mask = tokenizer.mask_token
                        input = context + choice + mask + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                elif pattern_class == "soft-prompt-cls":
                    if pattern_idx == "0":
                        # <s>question</s></s>triple</s></s>_ _ _ _ _ _Candidate answer is choice.
                        context = context
                        triple = triple
                        choice = "Candidate answer is " + ending + "."
                        soft_prompt = " ".join([prompt_token for i in range(num_prompt_token)])
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + \
                                " " + soft_prompt + " " + choice + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        # <s>question</s></s>triple</s></s>_ _ _ choice _ _ _ .
                        context = context
                        triple = triple
                        choice = ending
                        soft_prompt_1 = " ".join([prompt_token for i in range(0, int(num_prompt_token/2))])
                        soft_prompt_2 = " ".join([prompt_token for i in range(int(num_prompt_token/2), num_prompt_token)])
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + soft_prompt_1 + " " + choice + \
                                " " + soft_prompt_2 + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "2":
                        # <s>question</s></s>_ _ _ _ choice</s></s>triple</s> (initialized with hard prompt embeddings)
                        context = context
                        triple = triple
                        choice = ending
                        soft_prompt = " ".join([prompt_token for i in range(num_prompt_token)])
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                soft_prompt + choice + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                elif pattern_class == "soft-prompt-gen":
                    if pattern_idx == "0":
                        # <s>question</s></s>triple</s></s>Is choice the answer? _ _ _ _ _ _ [MASK].
                        context = context
                        triple = triple
                        choice = "Is " + ending + " the answer?"
                        mask = tokenizer.mask_token + "."
                        soft_prompt = " ".join([prompt_token for i in range(num_prompt_token)])
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + choice + \
                                soft_prompt + mask + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        # <s>question</s></s>triple</s></s>_ _ _ choice _ _ _ [MASK].
                        context = context
                        triple = triple
                        choice = ending
                        mask = tokenizer.mask_token + "."
                        soft_prompt = " ".join([prompt_token for i in range(int(num_prompt_token/2))])
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + soft_prompt + " " + choice + \
                                " " + soft_prompt + " " + mask
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "2":
                        # <s>question</s></s>triple</s></s>_ choice _ _ _ [MASK], it is. (initialized with hp)
                        context = context
                        triple = triple
                        choice = ending
                        mask = tokenizer.mask_token
                        soft_prompt_1 = " ".join([prompt_token for i in range(0,1)]) # is
                        soft_prompt_2 = " ".join([prompt_token for i in range(1, 4)]) # the answer?
                        soft_prompt_3 = " ".join([prompt_token for i in range(4, num_prompt_token)]) # , it is.
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + soft_prompt_1 + " " + choice + \
                                " " + soft_prompt_2 + " " + mask + " " + soft_prompt_3 + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "3":
                        # <s>_ _ _ _ _ _ _</s></s>question</s></s>triple</s></s>Is choice the answer?[MASK].
                        context = context
                        triple = triple
                        choice = "Is " + ending + " the answer?"
                        mask = tokenizer.mask_token + "."
                        soft_prompt = " ".join([prompt_token for i in range(num_prompt_token)])
                        input = tokenizer.cls_token + soft_prompt + tokenizer.sep_token + tokenizer.sep_token + \
                                context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + \
                                choice + mask + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)

                if ex_index == ending_idx == 0:
                    print("Tokenized tokens is:")
                    print(input_tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                    ori = tokenizer.decode(input_ids)
                    print("Using input pattern of %s "%(ori))


                # convert to ids
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                # not use segment_ids, only keep format consistent.
                segment_ids = [0] * len(input_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                block_flag = [0]*len(input_ids)
                mlm_mask = [0]*len(input_ids)
                if pattern_class == "no-prompt-cls":
                    if pattern_idx == "0":
                        mlm_mask[0] = 1
                elif pattern_class == "hard-prompt-cls":
                    mlm_mask[0] = 1
                elif pattern_class == "hard-prompt-gen":
                    mlm_mask[mask_token_index[0]] = 1 # 1 for masked token
                elif pattern_class == "soft-prompt-cls":
                    for idx in prompt_token_index:
                        block_flag[idx] = 1  # 1 for prompt placeholder
                    mlm_mask[0] = 1
                elif pattern_class == "soft-prompt-gen":
                    for idx in prompt_token_index:
                        block_flag[idx] = 1  # 1 for prompt placeholder
                    mlm_mask[mask_token_index[0]] = 1 # 1 for masked token


                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                block_flag = block_flag + ([0] * padding_length)
                mlm_mask = mlm_mask + ([0] * padding_length)


                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = 1 if ending_idx == label else 0
                choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label))
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/huggingface/roberta-large/'
    tokenizer = RobertaTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_albert_input_tensors_for_kcr(args, statement_jsonl_path, model_type, model_name, max_seq_length, num_prompt_token):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, triples, surfaces, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.triples = triples
            self.surfaces = surfaces
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            all = json.load(f)
            examples = []
            for json_dic in all:
                choices = json_dic['question']['choices']
                question_concept = json_dic['question']['question_concept']
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                context = json_dic["question"]["stem"]

                triples = []
                endings = []
                contexts = []
                surfaces = []
                for i in range(len(choices)):
                    choice = choices[i]
                    if choice['triple']:
                        triple_str = ' '.join(choice['triple'][0])
                        triples_temp = triple_str
                        surface = choice['surface']
                        surface_str = surface.replace('[', '').replace(']', '')
                    else:
                        triples_temp = question_concept + tokenizer.sep_token + tokenizer.sep_token + choice['text']
                        surface_str = ""
                    triples.append(triples_temp)
                    endings.append(choice['text'])
                    contexts.append(context)
                    surfaces.append(surface_str)
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=contexts,
                        question="",
                        endings=endings,
                        triples=triples,
                        surfaces=surfaces,
                        label=label
                    ))
        return examples

    def convert_examples_to_features(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        ''' Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet) '''
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            choices_features = []
            # alberta: [CLS] Question is A? [SEP] Answer is B. [SEP]
            # for one sample
            for ending_idx, (context, ending, triple, surface) in \
                    enumerate(zip(example.contexts, example.endings, example.triples, example.surfaces)):
                # no prompt cls
                if pattern_class == "hard-prompt-cls":
                    if pattern_idx == "2":
                        # <s>question</s></s>According to: triple</s></s>Candidate answer is choice.</s>
                        context = context
                        surface = surface
                        if surface != '':
                            choice = "According to " + surface.lower() + ". " + "The answer is: " + ending.lower() + "."
                        else:
                            choice = "The answer is: " + ending.lower() + "."
                        sent1 = context
                        sent2 = choice
                        input = tokenizer.cls_token + " " + sent1 + " " + tokenizer.sep_token + " " + \
                                sent2 + " " + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)

                if ex_index == ending_idx == 0:
                    print("Tokenized tokens is:")
                    print(input_tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                    ori = tokenizer.decode(input_ids)
                    print("Using input pattern of %s "%(ori))

                # convert to ids
                # input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                sent1_tokens = tokenizer.tokenize(sent1)
                sent2_tokens = tokenizer.tokenize(sent2)
                sent1_ids = tokenizer.convert_tokens_to_ids(sent1_tokens)
                sent2_ids = tokenizer.convert_tokens_to_ids(sent2_tokens)
                input_ids = tokenizer.build_inputs_with_special_tokens(sent1_ids, sent2_ids)
                # not use segment_ids, only keep format consistent.
                # segment_ids = [0] * len(input_ids)
                segment_ids = tokenizer.create_token_type_ids_from_sequences(sent1_ids, sent2_ids)
                # assert len(input_ids) == len(segment_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                block_flag = [0]*len(input_ids)
                mlm_mask = [0]*len(input_ids)
                if pattern_class == "hard-prompt-cls":
                    mlm_mask[0] = 1

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                block_flag = block_flag + ([0] * padding_length)
                mlm_mask = mlm_mask + ([0] * padding_length)


                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = 1 if ending_idx == label else 0
                choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label))
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/albert-xxlarge-v2/'
    tokenizer = AlbertTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_roberta_input_tensors_for_csqav2(args, statement_jsonl_path, model_type, model_name, max_seq_length, num_prompt_token):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, topic, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.topic = topic
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        label_map = {"no":0,"yes":1}
        with open(input_file, "r", encoding="utf-8") as f:
            all = f.readlines()
            examples = []
            for line in all:
                json_dic = json.loads(line)
                question = json_dic['question']
                label = label_map[json_dic["answer"]] if 'answer' in json_dic else 0
                choices = ['no','yes']
                topic = json_dic['topic_prompt'] if json_dic['topic_prompt_used'] else None
                contexts = [question]*len(choices)
                endings = choices
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=contexts,
                        question="",
                        endings=endings,
                        topic = topic,
                        label=label
                    ))
        return examples

    def convert_examples_to_features(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        # {0:0, 1:1}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        real_max_len = 0
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            topic = example.topic
            choices_features = []
            # roberta: <s>Question is A?</s></s>Answer is B.</s>
            # for one sample
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                if pattern_class == "hard-prompt-cls":
                    if pattern_idx == "0":
                        question = context.strip()
                        if not (question.endswith("?") or question.endswith(".")):
                            question = question + "."
                        if question.endswith("."):
                            question = "Is it true that {}?".format(question.lower()[:-1])
                        # question = "Question: {}".format(question)
                        sent1 = question
                        if topic is not None:
                            choice = "right" if ending=='yes' else "incorrect"
                            ending = ending[0].upper() + ending[1:]
                            sent2 = ending + ", the description about " + topic + " is " + choice + "."
                        else:
                            choice = "right" if ending == 'yes' else "incorrect"
                            ending = ending[0].upper() + ending[1:]
                            sent2 = ending + ", this description is " + choice + "."
                        input = tokenizer.cls_token + sent1 + tokenizer.sep_token + tokenizer.sep_token + \
                                sent2 + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        question = context.strip()
                        if not (question.endswith("?") or question.endswith(".")):
                            question = question + "."
                        if question.endswith("."):
                            question = "Is it true that {}?".format(question.lower()[:-1])
                        question = "Question: {}".format(question)
                        sent1 = question
                        choice = ending[0].upper() + ending[1:]
                        choice = choice + ", it is right." if choice=='Yes' else choice +", it is incorrect."
                        sent2 = "Answer: {}".format(choice)

                        input = tokenizer.cls_token + sent1 + tokenizer.sep_token + tokenizer.sep_token + \
                                sent2 + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)

                if ex_index == ending_idx == 0:
                    print("Tokenized tokens is:")
                    print(input_tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                    ori = tokenizer.decode(input_ids)
                    print("Using input pattern of %s "%(ori))


                # convert to ids
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                real_max_len = max(real_max_len, len(input_ids))
                # not use segment_ids, only keep format consistent.
                segment_ids = [0] * len(input_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                block_flag = [0]*len(input_ids)
                mlm_mask = [0]*len(input_ids)
                if pattern_class == "hard-prompt-cls":
                    mlm_mask[0] = 1

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                block_flag = block_flag + ([0] * padding_length)
                mlm_mask = mlm_mask + ([0] * padding_length)


                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = 1 if ending_idx == label else 0
                choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label))
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))
        print("Real max length is: %d, setting is: %d"%(real_max_len, max_seq_length))
        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/huggingface/roberta-large/'
    tokenizer = RobertaTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_albert_input_tensors_for_csqav2(args, statement_jsonl_path, model_type, model_name, max_seq_length, num_prompt_token):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, topic, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.topic = topic
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        label_map = {"no":0,"yes":1}
        with open(input_file, "r", encoding="utf-8") as f:
            all = f.readlines()
            examples = []
            for line in all:
                json_dic = json.loads(line)
                question = json_dic['question']
                label = label_map[json_dic["answer"]] if 'answer' in json_dic else 0
                choices = ['no','yes']
                topic = json_dic['topic_prompt'] if json_dic['topic_prompt_used'] else None
                contexts = [question]*len(choices)
                endings = choices
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=contexts,
                        question="",
                        endings=endings,
                        topic = topic,
                        label=label
                    ))
        return examples

    def convert_examples_to_features(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        # {0:0, 1:1}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        real_max_len = 0
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            topic = example.topic
            choices_features = []
            # roberta: <s>Question is A?</s></s>Answer is B.</s>
            # for one sample
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                if pattern_class == "hard-prompt-cls":
                    if pattern_idx == "0":
                        question = context.strip()
                        if not (question.endswith("?") or question.endswith(".")):
                            question = question + "."
                        if question.endswith("."):
                            question = "Is it true that {}?".format(question.lower()[:-1])
                        # question = "Question: {}".format(question)
                        sent1 = question
                        if topic is not None:
                            choice = "right" if ending=='yes' else "incorrect"
                            ending = ending[0].upper() + ending[1:]
                            sent2 = ending + ", the description about " + topic + " is " + choice + "."
                        else:
                            choice = "right" if ending == 'yes' else "incorrect"
                            ending = ending[0].upper() + ending[1:]
                            sent2 = ending + ", this description is " + choice + "."
                        input = tokenizer.cls_token + sent1 + tokenizer.sep_token + tokenizer.sep_token + \
                                sent2 + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)

                # convert to ids
                # input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                sent1_tokens = tokenizer.tokenize(sent1)
                sent2_tokens = tokenizer.tokenize(sent2)
                sent1_ids = tokenizer.convert_tokens_to_ids(sent1_tokens)
                sent2_ids = tokenizer.convert_tokens_to_ids(sent2_tokens)
                input_ids = tokenizer.build_inputs_with_special_tokens(sent1_ids, sent2_ids)
                if ex_index == ending_idx == 0:
                    ori = tokenizer.decode(input_ids)
                    print("Using input pattern of %s "%(ori))
                # not use segment_ids, only keep format consistent.
                # segment_ids = [0] * len(input_ids)
                segment_ids = tokenizer.create_token_type_ids_from_sequences(sent1_ids, sent2_ids)
                # assert len(input_ids) == len(segment_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                block_flag = [0]*len(input_ids)
                mlm_mask = [0]*len(input_ids)
                if pattern_class == "hard-prompt-cls":
                    mlm_mask[0] = 1

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                block_flag = block_flag + ([0] * padding_length)
                mlm_mask = mlm_mask + ([0] * padding_length)


                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = 1 if ending_idx == label else 0
                choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label))
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))
        print("Real max length is: %d, setting is: %d"%(real_max_len, max_seq_length))
        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/albert-xxlarge-v2/'
    tokenizer = AlbertTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)


def load_bert_input_tensors_for_kcr(args, statement_jsonl_path, model_type, model_name, max_seq_length, num_prompt_token):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, triples, surfaces, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.triples = triples
            self.surfaces = surfaces
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            all = json.load(f)
            examples = []
            for json_dic in all:
                choices = json_dic['question']['choices']
                question_concept = json_dic['question']['question_concept']
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                context = json_dic["question"]["stem"]

                triples = []
                endings = []
                contexts = []
                surfaces = []
                for i in range(len(choices)):
                    choice = choices[i]
                    if choice['triple']:
                        triple_str = ' '.join(choice['triple'][0])
                        triples_temp = triple_str
                        surface = choice['surface']
                        surface_str = surface.replace('[','').replace(']','')
                    else:
                        triples_temp = question_concept + tokenizer.sep_token + tokenizer.sep_token + choice['text']
                        surface_str = ""
                    triples.append(triples_temp)
                    endings.append(choice['text'])
                    contexts.append(context)
                    surfaces.append(surface_str)
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=contexts,
                        question="",
                        endings=endings,
                        triples=triples,
                        surfaces=surfaces,
                        label=label
                    ))
        return examples

    def convert_examples_to_features(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            choices_features = []
            cdx_list = ["A", "B", "C", "D", "E"]
            if pattern_class == "hard-prompt-gen" and pattern_idx == "1":
                # <s>question</s></s>c1 t1</s></s>c2 t2</s></s>c3 t3</s></s>Is the right answer c1. [MASK]</s>
                input = tokenizer.cls_token
                sep_token = tokenizer.sep_token
                for ending_idx, (context, ending, triple, cdx) in enumerate(
                        zip(example.contexts, example.endings, example.triples, cdx_list)):
                    choice_feature = ending + ": " + triple
                    if ending_idx == 0:
                        input = input + context + sep_token + sep_token + choice_feature
                    else:
                        input = input + sep_token + sep_token + choice_feature
                prefix = input + sep_token + sep_token
            # roberta: <s>Question is A?</s></s>Answer is B.</s>
            # for one sample
            for ending_idx, (context, ending, triple, surface) in \
                    enumerate(zip(example.contexts, example.endings, example.triples, example.surfaces)):
                # no prompt cls
                if pattern_class == "no-prompt-cls":
                    if pattern_idx == "0":
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + triple + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                elif pattern_class == "hard-prompt-cls":
                    if pattern_idx == "0":
                        # <s>question</s></s>Candidate answer is choice.</s></s>triple</s>
                        context = context
                        triple = triple
                        choice = "Candidate answer is " + ending + "."
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                choice + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        # <s>question</s></s>According to: triple</s></s>Candidate answer is choice.</s>
                        context = context
                        triple = "According to: " + triple + "."
                        choice = "Candidate answer is " + ending + "."
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + \
                                choice + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "2":
                        # <s>question</s></s>According to suface, the answer is choice</s></s>triple</s>
                        context = context
                        surface = surface
                        if surface!='':
                            choice = "According to " + surface.lower() + ". " + "The answer is: " + ending.lower() + "."
                        else:
                            choice = "The answer is: " + ending.lower() + "."
                        sent1 = context
                        sent2 = choice
                        input = tokenizer.cls_token + " " + sent1 + " " + tokenizer.sep_token + " " + \
                                sent2 + " " + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "3":
                        # <s>question</s></s>The answer is choice, because</s></s>triple</s>
                        context = context
                        surface = surface
                        if surface!='':
                            choice = "The answer is: " + ending.lower() + ", " + "because " + surface.lower() + "."
                        else:
                            choice = "The answer is: " + ending.lower() + "."
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                choice + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                elif pattern_class == "hard-prompt-gen":
                    if pattern_idx == "0":
                        # <s>question</s></s>According to triple</s></s>Is choice the answer? [MASK], it is.
                        context = context
                        triple = "According to " + triple
                        choice = "Is " + ending + " the answer?"
                        mask = tokenizer.mask_token + ", it is."
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + choice + \
                                " " + mask + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        # <s>question</s></s>c1 t1</s></s>c2 t2</s></s>c3 t3</s></s>According to the choices, is the right answer c1. [MASK]</s>
                        context = prefix
                        choice = "According to the choices, is " + ending + " the right answer?"
                        mask = tokenizer.mask_token
                        input = context + choice + mask + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                elif pattern_class == "soft-prompt-cls":
                    if pattern_idx == "0":
                        # <s>question</s></s>triple</s></s>_ _ _ _ _ _Candidate answer is choice.
                        context = context
                        triple = triple
                        choice = "Candidate answer is " + ending + "."
                        soft_prompt = " ".join([prompt_token for i in range(num_prompt_token)])
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + \
                                " " + soft_prompt + " " + choice + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        # <s>question</s></s>triple</s></s>_ _ _ choice _ _ _ .
                        context = context
                        triple = triple
                        choice = ending
                        soft_prompt_1 = " ".join([prompt_token for i in range(0, int(num_prompt_token/2))])
                        soft_prompt_2 = " ".join([prompt_token for i in range(int(num_prompt_token/2), num_prompt_token)])
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + soft_prompt_1 + " " + choice + \
                                " " + soft_prompt_2 + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "2":
                        # <s>question</s></s>_ _ _ _ choice</s></s>triple</s> (initialized with hard prompt embeddings)
                        context = context
                        triple = triple
                        choice = ending
                        soft_prompt = " ".join([prompt_token for i in range(num_prompt_token)])
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                soft_prompt + choice + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                elif pattern_class == "soft-prompt-gen":
                    if pattern_idx == "0":
                        # <s>question</s></s>triple</s></s>Is choice the answer? _ _ _ _ _ _ [MASK].
                        context = context
                        triple = triple
                        choice = "Is " + ending + " the answer?"
                        mask = tokenizer.mask_token + "."
                        soft_prompt = " ".join([prompt_token for i in range(num_prompt_token)])
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + choice + \
                                soft_prompt + mask + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        # <s>question</s></s>triple</s></s>_ _ _ choice _ _ _ [MASK].
                        context = context
                        triple = triple
                        choice = ending
                        mask = tokenizer.mask_token + "."
                        soft_prompt = " ".join([prompt_token for i in range(int(num_prompt_token/2))])
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + soft_prompt + " " + choice + \
                                " " + soft_prompt + " " + mask
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "2":
                        # <s>question</s></s>triple</s></s>_ choice _ _ _ [MASK], it is. (initialized with hp)
                        context = context
                        triple = triple
                        choice = ending
                        mask = tokenizer.mask_token
                        soft_prompt_1 = " ".join([prompt_token for i in range(0,1)]) # is
                        soft_prompt_2 = " ".join([prompt_token for i in range(1, 4)]) # the answer?
                        soft_prompt_3 = " ".join([prompt_token for i in range(4, num_prompt_token)]) # , it is.
                        input = tokenizer.cls_token + context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + soft_prompt_1 + " " + choice + \
                                " " + soft_prompt_2 + " " + mask + " " + soft_prompt_3 + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "3":
                        # <s>_ _ _ _ _ _ _</s></s>question</s></s>triple</s></s>Is choice the answer?[MASK].
                        context = context
                        triple = triple
                        choice = "Is " + ending + " the answer?"
                        mask = tokenizer.mask_token + "."
                        soft_prompt = " ".join([prompt_token for i in range(num_prompt_token)])
                        input = tokenizer.cls_token + soft_prompt + tokenizer.sep_token + tokenizer.sep_token + \
                                context + tokenizer.sep_token + tokenizer.sep_token + \
                                triple + tokenizer.sep_token + tokenizer.sep_token + \
                                choice + mask + tokenizer.sep_token
                        input_tokens = tokenizer.tokenize(input)

                if ex_index == ending_idx == 0:
                    print("Tokenized tokens is:")
                    print(input_tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                    ori = tokenizer.decode(input_ids)
                    print("Using input pattern of %s "%(ori))


                # convert to ids
                # input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                sent1_tokens = tokenizer.tokenize(sent1)
                sent2_tokens = tokenizer.tokenize(sent2)
                sent1_ids = tokenizer.convert_tokens_to_ids(sent1_tokens)
                sent2_ids = tokenizer.convert_tokens_to_ids(sent2_tokens)
                input_ids = tokenizer.build_inputs_with_special_tokens(sent1_ids, sent2_ids)
                # not use segment_ids, only keep format consistent.
                # segment_ids = [0] * len(input_ids)
                segment_ids = tokenizer.create_token_type_ids_from_sequences(sent1_ids, sent2_ids)
                # assert len(input_ids) == len(segment_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                block_flag = [0]*len(input_ids)
                mlm_mask = [0]*len(input_ids)
                if pattern_class == "no-prompt-cls":
                    if pattern_idx == "0":
                        mlm_mask[0] = 1
                elif pattern_class == "hard-prompt-cls":
                    mlm_mask[0] = 1
                elif pattern_class == "hard-prompt-gen":
                    mlm_mask[mask_token_index[0]] = 1 # 1 for masked token
                elif pattern_class == "soft-prompt-cls":
                    for idx in prompt_token_index:
                        block_flag[idx] = 1  # 1 for prompt placeholder
                    mlm_mask[0] = 1
                elif pattern_class == "soft-prompt-gen":
                    for idx in prompt_token_index:
                        block_flag[idx] = 1  # 1 for prompt placeholder
                    mlm_mask[mask_token_index[0]] = 1 # 1 for masked token


                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                block_flag = block_flag + ([0] * padding_length)
                mlm_mask = mlm_mask + ([0] * padding_length)


                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = 1 if ending_idx == label else 0
                choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label))
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/bert-large-cased/'
    tokenizer = BertTokenizer.from_pretrained(path)
    print("Load tokenizer from: ", path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_roberta_input_tensors_for_kcr_with_concate_choice(args, statement_jsonl_path, model_type, model_name, max_seq_length, num_prompt_token):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, triples, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.triples = triples
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            all = json.load(f)
            examples = []
            for json_dic in all:
                choices = json_dic['question']['choices']
                question_concept = json_dic['question']['question_concept']
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                context = json_dic["question"]["stem"]

                triples = []
                endings = []
                contexts = []
                for i in range(len(choices)):
                    choice = choices[i]
                    if choice['triple']:
                        triple_str = ' '.join(choice['triple'][0])
                        triples_temp = triple_str
                    else:
                        # triples_temp = question_concept + tokenizer.sep_token + tokenizer.sep_token + choice['text']
                        triples_temp = question_concept + " " + choice['text']

                    triples.append(triples_temp)
                    endings.append(choice['text'])
                    contexts.append(context)
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=contexts,
                        question="",
                        endings=endings,
                        triples=triples,
                        label=label
                    ))
        return examples

    def convert_examples_to_features_with_concat_choices(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        ''' Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet) '''
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        sep_token = tokenizer.sep_token
        cls_token = tokenizer.cls_token
        mask_token = tokenizer.mask_token
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            choices_features = []
            if pattern_class == "no-prompt-cls":
                if pattern_idx == "0":
                    # <s>question</s></s>c1 t1</s></s>c2 t2</s></s>c3 t3</s>
                    input = tokenizer.cls_token
                    sep_token = tokenizer.sep_token
                    for ending_idx, (context, ending, triple) in enumerate(
                        zip(example.contexts, example.endings, example.triples)):
                        choice_feature = ending + ": " + triple
                        if ending_idx == 0:
                            input = input + context + sep_token + sep_token + choice_feature
                        else:
                            input = input + sep_token + sep_token + choice_feature
                    input = input + sep_token
                    input_tokens = tokenizer.tokenize(input)
                elif pattern_idx == "1":
                    # <s>question</s></s>t1</s></s>t2</s></s>t3</s>
                    input = tokenizer.cls_token
                    sep_token = tokenizer.sep_token
                    for ending_idx, (context, ending, triple) in enumerate(
                        zip(example.contexts, example.endings, example.triples)):
                        choice_feature = triple
                        if ending_idx == 0:
                            input = input + context + sep_token + sep_token + choice_feature
                        else:
                            input = input + sep_token + sep_token + choice_feature
                    input = input + sep_token
                    input_tokens = tokenizer.tokenize(input)
            elif pattern_class == "hard-prompt-cls":
                if pattern_idx == "0":
                    # <s>question</s></s>Candidate answer: choice. Answer information: triple</s></s>
                    input = tokenizer.cls_token
                    sep_token = tokenizer.sep_token
                    for ending_idx, (context, ending, triple) in enumerate(
                            zip(example.contexts, example.endings, example.triples)):
                        choice_feature = "Candidate answer: " + ending + "." + " Answer information: " + triple + "."
                        if ending_idx == 0:
                            input = input + context + sep_token + sep_token + choice_feature
                        else:
                            input = input + sep_token + sep_token + choice_feature
                    input = input + sep_token
                    input_tokens = tokenizer.tokenize(input)
            elif pattern_class == "hard-prompt-gen":
                cdx_list = ["A","B","C","D","E"]
                if pattern_idx == "0":
                    # <s>question</s></s>A:c1 t1</s></s>B:c2 t2</s></s>C:c3 t3</s></s>The answer is [mask].</s>
                    input = tokenizer.cls_token
                    sep_token = tokenizer.sep_token
                    for ending_idx, (context, ending, triple, cdx) in enumerate(
                        zip(example.contexts, example.endings, example.triples, cdx_list)):
                        choice_feature = cdx + ": " + ending + ". " + triple
                        if ending_idx == 0:
                            input = input + context + sep_token + sep_token + choice_feature
                        else:
                            input = input + sep_token + sep_token + choice_feature
                    input = input + sep_token + sep_token + "The answer is " + tokenizer.mask_token + "." + sep_token
                    input_tokens = tokenizer.tokenize(input)

            if ex_index == 0:
                print("Tokenized tokens is:")
                print(input_tokens)
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                ori = tokenizer.decode(input_ids)
                print("Using input pattern of %s " % (ori))

            # convert to ids
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            # not use segment_ids, only keep format consistent.
            segment_ids = [0] * len(input_ids)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1] * len(input_ids)
            special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
            output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

            # get the mask position, the first is used for insert prompt, the second is used for prediction
            mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
            mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
            prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
            prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

            block_flag = [0] * len(input_ids)
            mlm_mask = [0] * len(input_ids)
            if pattern_class == "no-prompt-cls":
                if pattern_idx == "0":
                    sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                    sep_token_idx = []
                    for i, j in enumerate(input_ids):
                        if j in sep_token_id:
                            sep_token_idx.append(i)
                    sep_token_idx = sep_token_idx[1:-1]
                    if len(sep_token_idx) != 10:
                        print("The number of sep token is %d!"%(len(sep_token_idx)))
                        print("Tokenized tokens is:")
                        print(input_tokens)
                        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                        ori = tokenizer.decode(input_ids)
                        print("Using input pattern of %s " % (ori))
                    # each second </s> of </s></s> is chosen to represent the choice feature.
                    for i in range(1, len(sep_token_idx), 2):
                        mlm_mask[sep_token_idx[i]] = 1
                    if sum(mlm_mask) != 5:
                        print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))
                elif pattern_idx == "1":
                    sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                    sep_token_idx = []
                    for i, j in enumerate(input_ids):
                        if j in sep_token_id:
                            sep_token_idx.append(i)
                    sep_token_idx = sep_token_idx[1:-1]
                    if len(sep_token_idx) != 10:
                        print("The number of sep token is %d!"%(len(sep_token_idx)))
                        print("Tokenized tokens is:")
                        print(input_tokens)
                        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                        ori = tokenizer.decode(input_ids)
                        print("Using input pattern of %s " % (ori))
                    # each second </s> of </s></s> is chosen to represent the choice feature.
                    for i in range(1, len(sep_token_idx), 2):
                        mlm_mask[sep_token_idx[i]] = 1
                    if sum(mlm_mask) != 5:
                        print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))
            elif pattern_class == "hard-prompt-gen":
                if pattern_idx == "0":
                    mlm_mask[mask_token_index[0]] = 1
            elif pattern_class == "hard-prompt-cls":
                if pattern_idx == "0":
                    sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
                    sep_token_idx = []
                    for i, j in enumerate(input_ids):
                        if j in sep_token_id:
                            sep_token_idx.append(i)
                    sep_token_idx = sep_token_idx[:-1]
                    if len(sep_token_idx) != 10:
                        print("The number of sep token is %d!" % (len(sep_token_idx)))
                        print("Tokenized tokens is:")
                        print(input_tokens)
                        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                        ori = tokenizer.decode(input_ids)
                        print("Using input pattern of %s " % (ori))
                    # each second </s> of </s></s> is chosen to represent the choice feature.
                    for i in range(1, len(sep_token_idx), 2):
                        mlm_mask[sep_token_idx[i]] = 1
                    if sum(mlm_mask) != 5:
                        print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            output_mask = output_mask + ([1] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)
            block_flag = block_flag + ([0] * padding_length)
            mlm_mask = mlm_mask + ([0] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(output_mask) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(block_flag) == max_seq_length
            assert len(mlm_mask) == max_seq_length
            mlm_label = label
            choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag,
                                     mlm_mask, mlm_label))

            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/huggingface/roberta-large/'
    tokenizer = RobertaTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features_with_concat_choices(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_roberta_input_tensors_for_kcr_with_concate_two_choice(args, statement_jsonl_path, model_type, model_name,
                                                               max_seq_length, num_prompt_token, data_type):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, triples, surfaces, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.triples = triples
            self.surfaces = surfaces
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            all = json.load(f)
            examples = []
            for json_dic in all:
                choices = json_dic['question']['choices']
                question_concept = json_dic['question']['question_concept']
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                context = json_dic["question"]["stem"]

                triples = []
                endings = []
                contexts = []
                surfaces = []
                for i in range(len(choices)):
                    choice = choices[i]
                    if choice['triple']:
                        triple_str = ' '.join(choice['triple'][0])
                        triples_temp = triple_str
                        surface = choice['surface']
                        surface_str = surface.replace('[', '').replace(']', '')
                    else:
                        triples_temp = question_concept + tokenizer.sep_token + tokenizer.sep_token + choice['text']
                        surface_str = ""
                    triples.append(triples_temp)
                    endings.append(choice['text'])
                    contexts.append(context)
                    surfaces.append(surface_str)

                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=contexts,
                        question="",
                        endings=endings,
                        triples=triples,
                        surfaces=surfaces,
                        label=label
                    ))
        return examples

    def convert_examples_to_features_for_train(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        ''' Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet) '''
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        sep_token = tokenizer.sep_token
        cls_token = tokenizer.cls_token
        mask_token = tokenizer.mask_token
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            choices_features = []
            question, right_e, right_t, right_s = \
                example.contexts[label], example.endings[label], example.triples[label], example.surfaces[label]

            for ending_idx, (context, ending, triple, surface) in enumerate(
                    zip(example.contexts, example.endings, example.triples, example.surfaces)):

                if ending_idx == label:
                    continue

                if pattern_class == "no-prompt-cls":
                    if pattern_idx == "0":
                        input = cls_token + question + sep_token + sep_token # cls question sep sep
                        answer = right_e + ": " + right_t
                        choice = ending + ": " + triple
                        if random.random() < 0.5:
                            input = input + answer + sep_token + sep_token + choice + sep_token # cls question sep sep answer sep sep choice sep
                            binary_label = 0
                        else:
                            input = input + choice + sep_token + sep_token + answer + sep_token  # cls question sep sep answer sep sep choice sep
                            binary_label = 1
                        input_tokens = tokenizer.tokenize(input)
                elif pattern_class == "hard-prompt-cls":
                    if pattern_idx == "0":
                        # <s>question</s></s>Candidate answer: choice. Answer information: triple</s></s>
                        input = cls_token + question + sep_token + sep_token
                        answer = "Candidate answer: " + right_e + ". " + "Answer information: " + right_t + "."
                        choice = "Candidate answer: " + ending + ". " + "Answer information: " + triple + "."
                        if random.random() < 0.5:
                            input = input + answer + sep_token + sep_token + choice + sep_token # cls question sep sep answer sep sep choice sep
                            binary_label = 0
                        else:
                            input = input + choice + sep_token + sep_token + answer + sep_token  # cls question sep sep answer sep sep choice sep
                            binary_label = 1
                        input_tokens = tokenizer.tokenize(input)
                    elif pattern_idx == "1":
                        # <s>question</s></s>Candidate answer: choice. Answer information: triple</s></s>
                        input = cls_token + question + sep_token + sep_token
                        if random.random() < 0.5:
                            answer = "According to " + right_s.lower() + ", " + "the candidate answer is: " + right_e.lower() + "."
                            choice = "According to " + surface.lower() + ", " + "another candidate answer is: " + ending.lower() + "."
                            input = input + answer + sep_token + sep_token + choice + sep_token  # cls question sep sep answer sep sep choice sep
                            binary_label = 0
                        else:
                            answer = "According to " + right_s.lower() + ", " + "another candidate answer is: " + right_e.lower() + "."
                            choice = "According to " + surface.lower() + ", " + "the candidate answer is: " + ending.lower() + "."
                            input = input + choice + sep_token + sep_token + answer + sep_token  # cls question sep sep answer sep sep choice sep
                            binary_label = 1
                        input_tokens = tokenizer.tokenize(input)

                if ex_index == 0:
                    print("Tokenized tokens is:")
                    print(input_tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                    ori = tokenizer.decode(input_ids)
                    print("Using input pattern of %s " % (ori))

                # convert to ids
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                # not use segment_ids, only keep format consistent.
                segment_ids = [0] * len(input_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                block_flag = [0] * len(input_ids)
                mlm_mask = [0] * len(input_ids)
                if pattern_class == "no-prompt-cls":
                    if pattern_idx == "0": # cls question sep sep answer sep sep choice sep
                        sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
                        sep_token_idx = []
                        for i, j in enumerate(input_ids):
                            if j in sep_token_id:
                                sep_token_idx.append(i)
                        sep_token_idx = sep_token_idx[:-1]
                        if len(sep_token_idx) != 4:
                            print("The number of sep token is %d!"%(len(sep_token_idx)))
                            print("Tokenized tokens is:")
                            print(input_tokens)
                            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                            ori = tokenizer.decode(input_ids)
                            print("Using input pattern of %s " % (ori))
                        # each second </s> of </s></s> is chosen to represent the choice feature.
                        for i in range(1, len(sep_token_idx), 2):
                            mlm_mask[sep_token_idx[i]] = 1
                        if sum(mlm_mask) != 2:
                            print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))
                elif pattern_class == "hard-prompt-cls":
                    # cls question sep sep answer sep sep choice sep
                    sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
                    sep_token_idx = []
                    for i, j in enumerate(input_ids):
                        if j in sep_token_id:
                            sep_token_idx.append(i)
                    sep_token_idx = sep_token_idx[:-1]
                    if len(sep_token_idx) != 4:
                        print("The number of sep token is %d!" % (len(sep_token_idx)))
                        print("Tokenized tokens is:")
                        print(input_tokens)
                        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                        ori = tokenizer.decode(input_ids)
                        print("Using input pattern of %s " % (ori))
                    # each second </s> of </s></s> is chosen to represent the choice feature.
                    for i in range(1, len(sep_token_idx), 2):
                        mlm_mask[sep_token_idx[i]] = 1
                    if sum(mlm_mask) != 2:
                        print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))
                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                block_flag = block_flag + ([0] * padding_length)
                mlm_mask = mlm_mask + ([0] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = binary_label
                choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag,
                                         mlm_mask, mlm_label))

            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def convert_examples_to_features_for_eval(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        ''' Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet) '''
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        sep_token = tokenizer.sep_token
        cls_token = tokenizer.cls_token
        mask_token = tokenizer.mask_token
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            choices_features = []
            choices_list = []
            for ending_idx, (context, ending, triple, surface) in enumerate(
                    zip(example.contexts, example.endings, example.triples, example.surfaces)):
                question = context
                choice = (ending, triple, surface)
                choices_list.append(choice)

            for c1 in range(5):
                for c2 in range(5):
                    if c2 > c1:
                        if pattern_class == "no-prompt-cls":
                            if pattern_idx == "0":
                                input = cls_token + question + sep_token + sep_token # cls question sep sep
                                choice1 = choices_list[c1][0] + ": " + choices_list[c1][1]
                                choice2 = choices_list[c2][0] + ": " + choices_list[c2][1]
                                input = input + choice1 + sep_token + sep_token + choice2 + sep_token # cls question sep sep answer sep sep choice sep
                                input_tokens = tokenizer.tokenize(input)
                        if pattern_class == "hard-prompt-cls":
                            if pattern_idx == "0":
                                # <s>question</s></s>Candidate answer: choice. Answer information: triple</s></s>
                                input = cls_token + question + sep_token + sep_token
                                choice1 = "Candidate answer: " + choices_list[c1][0] + ". " + "Answer information: " + choices_list[c1][1] + "."
                                choice2 = "Candidate answer: " + choices_list[c2][0] + ". " + "Answer information: " + choices_list[c2][1] + "."
                                input = input + choice1 + sep_token + sep_token + choice2 + sep_token
                                input_tokens = tokenizer.tokenize(input)
                            elif pattern_idx == "1":
                                # <s>question</s></s>Candidate answer: choice. Answer information: triple</s></s>
                                input = cls_token + question + sep_token + sep_token
                                choice1 = "According to " + choices_list[c1][2].lower() + ", " + "the candidate answer is: " + choices_list[c1][0].lower() + "."
                                choice2 = "According to " + choices_list[c2][2].lower() + ", " + "another candidate answer is: " + choices_list[c2][0].lower() + "."
                                input = input + choice1 + sep_token + sep_token + choice2 + sep_token  # cls question sep sep answer sep sep choice sep
                                input_tokens = tokenizer.tokenize(input)

                        if ex_index == 0:
                            print("Tokenized tokens is:")
                            print(input_tokens)
                            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                            ori = tokenizer.decode(input_ids)
                            print("Using input pattern of %s " % (ori))

                        # convert to ids
                        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                        # not use segment_ids, only keep format consistent.
                        segment_ids = [0] * len(input_ids)

                        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                        input_mask = [1] * len(input_ids)
                        special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                        output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                        # get the mask position, the first is used for insert prompt, the second is used for prediction
                        mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                        mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                        prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                        prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                        block_flag = [0] * len(input_ids)
                        mlm_mask = [0] * len(input_ids)
                        if pattern_class == "no-prompt-cls":
                            if pattern_idx == "0": # cls question sep sep answer sep sep choice sep
                                sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
                                sep_token_idx = []
                                for i, j in enumerate(input_ids):
                                    if j in sep_token_id:
                                        sep_token_idx.append(i)
                                sep_token_idx = sep_token_idx[:-1]
                                if len(sep_token_idx) != 4:
                                    print("The number of sep token is %d!"%(len(sep_token_idx)))
                                    print("Tokenized tokens is:")
                                    print(input_tokens)
                                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                                    ori = tokenizer.decode(input_ids)
                                    print("Using input pattern of %s " % (ori))
                                # each second </s> of </s></s> is chosen to represent the choice feature.
                                for i in range(1, len(sep_token_idx), 2):
                                    mlm_mask[sep_token_idx[i]] = 1
                                if sum(mlm_mask) != 2:
                                    print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))
                        elif pattern_class == "hard-prompt-cls":
                            # cls question sep sep answer sep sep choice sep
                            sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
                            sep_token_idx = []
                            for i, j in enumerate(input_ids):
                                if j in sep_token_id:
                                    sep_token_idx.append(i)
                            sep_token_idx = sep_token_idx[:-1]
                            if len(sep_token_idx) != 4:
                                print("The number of sep token is %d!"%(len(sep_token_idx)))
                                print("Tokenized tokens is:")
                                print(input_tokens)
                                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                                ori = tokenizer.decode(input_ids)
                                print("Using input pattern of %s " % (ori))
                            # each second </s> of </s></s> is chosen to represent the choice feature.
                            for i in range(1, len(sep_token_idx), 2):
                                mlm_mask[sep_token_idx[i]] = 1
                            if sum(mlm_mask) != 2:
                                print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))

                        # Zero-pad up to the sequence length.
                        padding_length = max_seq_length - len(input_ids)
                        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                        input_mask = input_mask + ([0] * padding_length)
                        output_mask = output_mask + ([1] * padding_length)
                        segment_ids = segment_ids + ([0] * padding_length)
                        block_flag = block_flag + ([0] * padding_length)
                        mlm_mask = mlm_mask + ([0] * padding_length)

                        assert len(input_ids) == max_seq_length
                        assert len(output_mask) == max_seq_length
                        assert len(input_mask) == max_seq_length
                        assert len(segment_ids) == max_seq_length
                        assert len(block_flag) == max_seq_length
                        assert len(mlm_mask) == max_seq_length
                        mlm_label = label
                        choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag,
                                                 mlm_mask, mlm_label))

            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/huggingface/roberta-large/'
    tokenizer = RobertaTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    if data_type == "train":
        features = convert_examples_to_features_for_train(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)
    else:
        features = convert_examples_to_features_for_eval(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)

    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_albert_input_tensors_for_kcr_with_concate_two_choice(args, statement_jsonl_path, model_type, model_name,
                                                               max_seq_length, num_prompt_token, data_type):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, triples, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.triples = triples
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            all = json.load(f)
            examples = []
            for json_dic in all:
                choices = json_dic['question']['choices']
                question_concept = json_dic['question']['question_concept']
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                context = json_dic["question"]["stem"]

                triples = []
                endings = []
                contexts = []
                for i in range(len(choices)):
                    choice = choices[i]
                    if choice['triple']:
                        triple_str = ' '.join(choice['triple'][0])
                        triples_temp = triple_str
                    else:
                        # triples_temp = question_concept + tokenizer.sep_token + tokenizer.sep_token + choice['text']
                        triples_temp = question_concept + " " + choice['text']

                    triples.append(triples_temp)
                    endings.append(choice['text'])
                    contexts.append(context)
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=contexts,
                        question="",
                        endings=endings,
                        triples=triples,
                        label=label
                    ))
        return examples

    def convert_examples_to_features_for_train(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        ''' Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet) '''
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        sep_token = tokenizer.sep_token
        cls_token = tokenizer.cls_token
        mask_token = tokenizer.mask_token
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            choices_features = []
            question, right_e, right_t = example.contexts[label], example.endings[label], example.triples[label]
            for ending_idx, (context, ending, triple) in enumerate(
                    zip(example.contexts, example.endings, example.triples)):

                if ending_idx == label:
                    continue

                if pattern_class == "no-prompt-cls":
                    if pattern_idx == "0":
                        input = cls_token + question + sep_token # cls question sep sep
                        answer = right_e + ": " + right_t
                        choice = ending + ": " + triple
                        if random.random() < 0.5:
                            input = input + answer + sep_token + choice + sep_token # cls question sep sep answer sep sep choice sep
                            binary_label = 0
                        else:
                            input = input + choice + sep_token + answer + sep_token  # cls question sep sep answer sep sep choice sep
                            binary_label = 1
                        input_tokens = tokenizer.tokenize(input)
                elif pattern_class == "hard-prompt-cls":
                    if pattern_idx == "0":
                        # <s>question</s></s>Candidate answer: choice. Answer information: triple</s></s>
                        input = cls_token + question + sep_token
                        answer = "Candidate answer: " + right_e + ". " + "Answer information: " + right_t + "."
                        choice = "Candidate answer: " + ending + ". " + "Answer information: " + triple + "."
                        if random.random() < 0.5:
                            input = input + answer + sep_token + choice + sep_token # cls question sep sep answer sep sep choice sep
                            binary_label = 0
                        else:
                            input = input + choice + sep_token + answer + sep_token  # cls question sep sep answer sep sep choice sep
                            binary_label = 1
                        input_tokens = tokenizer.tokenize(input)

                if ex_index == 0:
                    print("Tokenized tokens is:")
                    print(input_tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                    ori = tokenizer.decode(input_ids)
                    print("Using input pattern of %s " % (ori))

                # convert to ids
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                # not use segment_ids, only keep format consistent.
                segment_ids = [0] * len(input_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # get the mask position, the first is used for insert prompt, the second is used for prediction
                mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                block_flag = [0] * len(input_ids)
                mlm_mask = [0] * len(input_ids)
                if pattern_class == "no-prompt-cls":
                    if pattern_idx == "0": # cls question sep answer sep choice sep
                        sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
                        sep_token_idx = []
                        for i, j in enumerate(input_ids):
                            if j in sep_token_id:
                                sep_token_idx.append(i)
                        sep_token_idx = sep_token_idx[:-1]
                        if len(sep_token_idx) != 2:
                            print("The number of sep token is %d!"%(len(sep_token_idx)))
                            print("Tokenized tokens is:")
                            print(input_tokens)
                            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                            ori = tokenizer.decode(input_ids)
                            print("Using input pattern of %s " % (ori))
                        # each second </s> of </s></s> is chosen to represent the choice feature.
                        for i in range(len(sep_token_idx)):
                            mlm_mask[sep_token_idx[i]] = 1
                        if sum(mlm_mask) != 2:
                            print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))
                elif pattern_class == "hard-prompt-cls":
                    if pattern_idx == "0": # cls question sep answer sep choice sep
                        sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
                        sep_token_idx = []
                        for i, j in enumerate(input_ids):
                            if j in sep_token_id:
                                sep_token_idx.append(i)
                        sep_token_idx = sep_token_idx[:-1]
                        if len(sep_token_idx) != 2:
                            print("The number of sep token is %d!"%(len(sep_token_idx)))
                            print("Tokenized tokens is:")
                            print(input_tokens)
                            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                            ori = tokenizer.decode(input_ids)
                            print("Using input pattern of %s " % (ori))
                        # each second </s> of </s></s> is chosen to represent the choice feature.
                        for i in range(len(sep_token_idx)):
                            mlm_mask[sep_token_idx[i]] = 1
                        if sum(mlm_mask) != 2:
                            print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([0] * padding_length)
                output_mask = output_mask + ([1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                block_flag = block_flag + ([0] * padding_length)
                mlm_mask = mlm_mask + ([0] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(block_flag) == max_seq_length
                assert len(mlm_mask) == max_seq_length
                mlm_label = binary_label
                choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag,
                                         mlm_mask, mlm_label))

            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def convert_examples_to_features_for_eval(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        ''' Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet) '''
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        sep_token = tokenizer.sep_token
        cls_token = tokenizer.cls_token
        mask_token = tokenizer.mask_token
        for ex_index, example in enumerate(examples):
            label = label_map[example.label]
            choices_features = []
            choices_list = []
            for ending_idx, (context, ending, triple) in enumerate(
                    zip(example.contexts, example.endings, example.triples)):
                question = context
                choice = (ending, triple)
                choices_list.append(choice)

            for c1 in range(5):
                for c2 in range(5):
                    if c2 > c1:
                        if pattern_class == "no-prompt-cls":
                            if pattern_idx == "0":
                                input = cls_token + question + sep_token # cls question sep sep
                                choice1 = choices_list[c1][0] + ": " + choices_list[c1][1]
                                choice2 = choices_list[c2][0] + ": " + choices_list[c2][1]
                                input = input + choice1 + sep_token + choice2 + sep_token # cls question sep sep answer sep sep choice sep
                                input_tokens = tokenizer.tokenize(input)
                        elif pattern_class == "hard-prompt-cls":
                            if pattern_idx == "0":
                                # <s>question</s></s>Candidate answer: choice. Answer information: triple</s></s>
                                input = cls_token + question + sep_token
                                choice1 = "Candidate answer: " + choices_list[c1][0] + ". " + "Answer information: " + choices_list[c1][1] + "."
                                choice2 = "Candidate answer: " + choices_list[c2][0] + ". " + "Answer information: " + choices_list[c2][1] + "."
                                input = input + choice1 + sep_token + choice2 + sep_token
                                input_tokens = tokenizer.tokenize(input)

                        if ex_index == 0:
                            print("Tokenized tokens is:")
                            print(input_tokens)
                            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                            ori = tokenizer.decode(input_ids)
                            print("Using input pattern of %s " % (ori))

                        # convert to ids
                        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                        # not use segment_ids, only keep format consistent.
                        segment_ids = [0] * len(input_ids)

                        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                        input_mask = [1] * len(input_ids)
                        special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
                        output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                        # get the mask position, the first is used for insert prompt, the second is used for prediction
                        mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
                        mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
                        prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
                        prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

                        block_flag = [0] * len(input_ids)
                        mlm_mask = [0] * len(input_ids)
                        if pattern_class == "no-prompt-cls":
                            if pattern_idx == "0": # cls question sep answer sep choice sep
                                sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
                                sep_token_idx = []
                                for i, j in enumerate(input_ids):
                                    if j in sep_token_id:
                                        sep_token_idx.append(i)
                                sep_token_idx = sep_token_idx[:-1]
                                if len(sep_token_idx) != 2:
                                    print("The number of sep token is %d!"%(len(sep_token_idx)))
                                    print("Tokenized tokens is:")
                                    print(input_tokens)
                                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                                    ori = tokenizer.decode(input_ids)
                                    print("Using input pattern of %s " % (ori))
                                # each second </s> of </s></s> is chosen to represent the choice feature.
                                for i in range(len(sep_token_idx)):
                                    mlm_mask[sep_token_idx[i]] = 1
                                if sum(mlm_mask) != 2:
                                    print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))
                        elif pattern_class == "hard-prompt-cls":
                            if pattern_idx == "0": # cls question sep answer sep choice sep
                                sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
                                sep_token_idx = []
                                for i, j in enumerate(input_ids):
                                    if j in sep_token_id:
                                        sep_token_idx.append(i)
                                sep_token_idx = sep_token_idx[:-1]
                                if len(sep_token_idx) != 2:
                                    print("The number of sep token is %d!"%(len(sep_token_idx)))
                                    print("Tokenized tokens is:")
                                    print(input_tokens)
                                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                                    ori = tokenizer.decode(input_ids)
                                    print("Using input pattern of %s " % (ori))
                                # each second </s> of </s></s> is chosen to represent the choice feature.
                                for i in range(len(sep_token_idx)):
                                    mlm_mask[sep_token_idx[i]] = 1
                                if sum(mlm_mask) != 2:
                                    print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))

                        # Zero-pad up to the sequence length.
                        padding_length = max_seq_length - len(input_ids)
                        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                        input_mask = input_mask + ([0] * padding_length)
                        output_mask = output_mask + ([1] * padding_length)
                        segment_ids = segment_ids + ([0] * padding_length)
                        block_flag = block_flag + ([0] * padding_length)
                        mlm_mask = mlm_mask + ([0] * padding_length)

                        assert len(input_ids) == max_seq_length
                        assert len(output_mask) == max_seq_length
                        assert len(input_mask) == max_seq_length
                        assert len(segment_ids) == max_seq_length
                        assert len(block_flag) == max_seq_length
                        assert len(mlm_mask) == max_seq_length
                        mlm_label = label
                        choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag,
                                                 mlm_mask, mlm_label))

            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/albert-xxlarge-v2/'
    tokenizer = AlbertTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    if data_type == "train":
        features = convert_examples_to_features_for_train(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)
    else:
        features = convert_examples_to_features_for_eval(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)

    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_albert_input_tensors_for_kcr_with_concate_choice(args, statement_jsonl_path, model_type, model_name, max_seq_length, num_prompt_token):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, triples, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.triples = triples
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'block_flag': block_flag,
                    'mlm_mask': mlm_mask,
                    'mlm_label': mlm_label,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, block_flag, mlm_mask, mlm_label in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            all = json.load(f)
            examples = []
            for json_dic in all:
                choices = json_dic['question']['choices']
                question_concept = json_dic['question']['question_concept']
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                context = json_dic["question"]["stem"]

                triples = []
                endings = []
                contexts = []
                for i in range(len(choices)):
                    choice = choices[i]
                    if choice['triple']:
                        triple_str = ' '.join(choice['triple'][0])
                        triples_temp = triple_str
                    else:
                        # triples_temp = question_concept + tokenizer.sep_token + tokenizer.sep_token + choice['text']
                        triples_temp = question_concept + " " + choice['text']

                    triples.append(triples_temp)
                    endings.append(choice['text'])
                    contexts.append(context)
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=contexts,
                        question="",
                        endings=endings,
                        triples=triples,
                        label=label
                    ))
        return examples

    def convert_examples_to_features_with_concat_choices(pattern_type, examples, label_list, max_seq_length, tokenizer, prompt_token="PROMPT"):
        ''' Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet) '''
        # {0:0, 1:1, 2:2, 3:3, 4:4}
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        pattern_class, pattern_idx = pattern_type.split("_")
        for ex_index, example in enumerate(examples):
            label = label_map[example.label] # label: 0,1,2,3,4
            choices_features = []
            if pattern_class == "no-prompt-cls":
                if pattern_idx == "0":
                    # [cls] question [SEP] c1 t1 [SEP] c2 t2 [SEP] c3 t3 [SEP]
                    # 0 0 0 1 1 1 0 0 0 1 1 1
                    input = tokenizer.cls_token
                    sep_token = tokenizer.sep_token
                    for ending_idx, (context, ending, triple) in enumerate(
                        zip(example.contexts, example.endings, example.triples)):
                        choice_feature = ending + ". " + triple
                        if ending_idx == 0:
                            input = input + " " + context + " " + sep_token + " " + choice_feature
                        else:
                            input = input + " " + sep_token + " " + choice_feature
                    input = input + " " + sep_token
                    input_tokens = tokenizer.tokenize(input)
            elif pattern_class == "hard-prompt-gen":
                cdx_list = ["A","B","C","D","E"]
                if pattern_idx == "0":
                    # <s>question</s></s>A:c1 t1</s></s>B:c2 t2</s></s>C:c3 t3</s></s>The answer is [mask].</s>
                    input = tokenizer.cls_token
                    sep_token = tokenizer.sep_token
                    for ending_idx, (context, ending, triple, cdx) in enumerate(
                        zip(example.contexts, example.endings, example.triples, cdx_list)):
                        choice_feature = cdx + ": " + ending + ". " + triple
                        if ending_idx == 0:
                            input = input + context + sep_token + sep_token + choice_feature
                        else:
                            input = input + sep_token + sep_token + choice_feature
                    input = input + sep_token + sep_token + "The answer is " + tokenizer.mask_token + "." + sep_token
                    input_tokens = tokenizer.tokenize(input)

            if ex_index == 0:
                print("Tokenized tokens is:")
                print(input_tokens)
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                ori = tokenizer.decode(input_ids)
                print("Using input pattern of %s " % (ori))

            # convert to ids
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            # not use segment_ids, only keep format consistent.
            segment_ids = [1] * len(input_ids)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1] * len(input_ids)
            special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token])
            output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

            # get the mask position, the first is used for insert prompt, the second is used for prediction
            mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])
            mask_token_index = [index for index, id in enumerate(input_ids) if id in mask_token_id]
            prompt_token_id = tokenizer.convert_tokens_to_ids([prompt_token])
            prompt_token_index = [index for index, id in enumerate(input_ids) if id in prompt_token_id]

            block_flag = [0] * len(input_ids)
            mlm_mask = [0] * len(input_ids)
            if pattern_class == "no-prompt-cls":
                if pattern_idx == "0":
                    special_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
                    sep_token_idx = []
                    for i, j in enumerate(input_ids):
                        if j in special_token_id:
                            sep_token_idx.append(i)

                    # start = 0
                    # for i, j in enumerate(sep_token_idx):
                    #     if i%2 == 0:
                    #         for k in range(start,j):
                    #             segment_ids[k] = 0
                    #     else:
                    #         for k in range(start, j):
                    #             segment_ids[k] = 1
                    #     start = j

                    sep_token_idx = sep_token_idx[:-1]
                    if len(sep_token_idx) != 5:
                        print("The number of sep token is %d!"%(len(sep_token_idx)))
                        print("Tokenized tokens is:")
                        print(input_tokens)
                        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                        ori = tokenizer.decode(input_ids)
                        print("Using input pattern of %s " % (ori))
                    # each second </s> of </s></s> is chosen to represent the choice feature.
                    for i in sep_token_idx:
                        mlm_mask[i] = 1
                    if sum(mlm_mask) != 5:
                        print("The number of sep token in mlm_mask is %d!" % (sum(mlm_mask)))
            elif pattern_class == "hard-prompt-gen":
                if pattern_idx == "0":
                    mlm_mask[mask_token_index[0]] = 1

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            output_mask = output_mask + ([1] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)
            block_flag = block_flag + ([0] * padding_length)
            mlm_mask = mlm_mask + ([0] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(output_mask) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(block_flag) == max_seq_length
            assert len(mlm_mask) == max_seq_length
            mlm_label = label
            choices_features.append((input_tokens, input_ids, input_mask, segment_ids, output_mask, block_flag,
                                     mlm_mask, mlm_label))

            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # (bs, 5, max_len)
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_block_flag = torch.tensor(select_field(features, 'block_flag'), dtype=torch.long)
        all_mlm_mask = torch.tensor(select_field(features, 'mlm_mask'), dtype=torch.long)
        # (bs, 5)
        all_mlm_label = torch.tensor(select_field(features, 'mlm_label'), dtype=torch.long)
        # (bs)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label, (all_block_flag, all_mlm_mask, all_mlm_label)

    path = '/mnt/nlp_model/albert-xxlarge-v2/'
    tokenizer = AlbertTokenizer.from_pretrained(path)
    prompt_token = '[PROMPT]'
    tokenizer.add_tokens([prompt_token])
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features_with_concat_choices(args.pattern_format, examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer, prompt_token=prompt_token)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label, prompt_data_tensors = convert_features_to_tensors(features)
    assert len(prompt_data_tensors) == 3, "Prompt data tensor error"
    return (example_ids, all_label, *data_tensors, prompt_data_tensors)

def load_lstm_input_tensors(input_jsonl_path, max_seq_length):
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        while len(tokens_a) + len(tokens_b) > max_length:
            tokens_a.pop() if len(tokens_a) > len(tokens_b) else tokens_b.pop()

    tokenizer = WordTokenizer.from_pretrained('lstm')
    qids, labels, input_ids, input_lengths = [], [], [], []
    pad_id, = tokenizer.convert_tokens_to_ids([PAD_TOK])
    with open(input_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            input_json = json.loads(line)
            qids.append(input_json['id'])
            labels.append(ord(input_json.get("answerKey", "A")) - ord("A"))
            instance_input_ids, instance_input_lengths = [], []
            question_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_json["question"]["stem"]))
            for ending in input_json["question"]["choices"]:
                question_ids_copy = question_ids.copy()
                answer_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ending["text"]))
                _truncate_seq_pair(question_ids_copy, answer_ids, max_seq_length)
                ids = question_ids_copy + answer_ids + [pad_id] * (max_seq_length - len(question_ids_copy) - len(answer_ids))
                instance_input_ids.append(ids)
                instance_input_lengths.append(len(question_ids_copy) + len(answer_ids))
            input_ids.append(instance_input_ids)
            input_lengths.append(instance_input_lengths)
    labels = torch.tensor(labels, dtype=torch.long)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    return qids, labels, input_ids, input_lengths

def load_input_tensors(args, input_jsonl_path, model_type, model_name, max_seq_length):
    if model_type in ('lstm',):
        return load_lstm_input_tensors(input_jsonl_path, max_seq_length)
    elif model_type in ('gpt',):
        return load_gpt_input_tensors(model_name, args.pattern_type, input_jsonl_path, max_seq_length, args.prompt_token_num)
    elif model_type in ('roberta'):
        return load_roberta_input_tensors(args, input_jsonl_path, model_type, model_name, max_seq_length, args.prompt_token_num)
    elif model_type in ('bert', 'xlnet', 'roberta', 'albert'):
        return load_bert_xlnet_roberta_input_tensors(args, input_jsonl_path, model_type, model_name, max_seq_length)

def load_input_tensors_for_kcr(args, input_jsonl_path, model_type, model_name, max_seq_length, data_type):
    if model_type == 'roberta':
        print("Using model type of ", model_type)
        if args.concat_choices:
            return load_roberta_input_tensors_for_kcr_with_concate_choice(args, input_jsonl_path, model_type,
                                                                          model_name, max_seq_length,
                                                                          args.prompt_token_num)
        elif args.concat_two_choices:
            return load_roberta_input_tensors_for_kcr_with_concate_two_choice(args, input_jsonl_path, model_type,
                                                                              model_name, max_seq_length,
                                                                              args.prompt_token_num, data_type)
        else:
            return load_roberta_input_tensors_for_kcr(args, input_jsonl_path, model_type, model_name, max_seq_length,
                                                      args.prompt_token_num)
    elif model_type == "albert":
        print("Using model type of ", model_type)
        if args.concat_choices:
            return load_albert_input_tensors_for_kcr_with_concate_choice(args, input_jsonl_path, model_type, model_name,
                                                                         max_seq_length, args.prompt_token_num)
        elif args.concat_two_choices:
            return load_albert_input_tensors_for_kcr_with_concate_two_choice(args, input_jsonl_path, model_type,
                                                                              model_name, max_seq_length,
                                                                              args.prompt_token_num, data_type)
        else:
            return load_albert_input_tensors_for_kcr(args, input_jsonl_path, model_type, model_name, max_seq_length,
                                                     args.prompt_token_num)
    elif model_type == 'bert':
        print("Using model type of ", model_type)
        if args.concat_choices:
            print(1)
            return load_roberta_input_tensors_for_kcr_with_concate_choice(args, input_jsonl_path, model_type,
                                                                          model_name, max_seq_length,
                                                                          args.prompt_token_num)
        elif args.concat_two_choices:
            print(2)
            return load_roberta_input_tensors_for_kcr_with_concate_two_choice(args, input_jsonl_path, model_type,
                                                                              model_name, max_seq_length,
                                                                              args.prompt_token_num, data_type)
        else:
            print(3)
            return load_bert_input_tensors_for_kcr(args, input_jsonl_path, model_type, model_name, max_seq_length,
                                                      args.prompt_token_num)

def load_input_tensors_for_csqav2(args, input_jsonl_path, model_type, model_name, max_seq_length, data_type):
    if model_type == 'roberta':
        print("Using model type of ", model_type)
        return load_roberta_input_tensors_for_csqav2(args, input_jsonl_path, model_type, model_name, max_seq_length,
                                                      args.prompt_token_num)
    elif model_type == "albert":
        print("Using model type of ", model_type)
        return load_albert_input_tensors_for_csqav2(args, input_jsonl_path, model_type, model_name, max_seq_length,
                                                     args.prompt_token_num)
    elif model_type == 'bert':
        print("Using model type of ", model_type)
        return load_bert_input_tensors_for_csqav2(args, input_jsonl_path, model_type, model_name, max_seq_length,
                                                      args.prompt_token_num)


def load_input_tensors_for_nli(args, input_jsonl_path, model_class, model_name, max_seq_length, data_type, num_label, cache):
    if model_class in ('roberta'):
        return load_roberta_input_tensors_for_nli(args, input_jsonl_path, model_class, model_name, max_seq_length,
                                                  args.prompt_token_num, num_label=num_label, cache=cache)
    elif model_class in ("albert"):
        return load_albert_input_tensors_for_nli(args, input_jsonl_path, model_class, model_name, max_seq_length,
                                                  args.prompt_token_num, num_label=num_label, cache=cache)

def load_info(statement_path: str):
    n = sum(1 for _ in open(statement_path, "r"))
    num_choice = None
    with open(statement_path, "r", encoding="utf-8") as fin:
        ids = []
        labels = []
        for line in fin:
            input_json = json.loads(line)
            labels.append(ord(input_json.get("answerKey", "A")) - ord("A"))
            ids.append(input_json['id'])
            if num_choice is None:
                num_choice = len(input_json["question"]["choices"])
        labels = torch.tensor(labels, dtype=torch.long)

    return ids, labels, num_choice

def load_statement_dict(statement_path):
    all_dict = {}
    with open(statement_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            instance_dict = json.loads(line)
            qid = instance_dict['id']
            all_dict[qid] = {
                'question': instance_dict['question']['stem'],
                'answers': [dic['text'] for dic in instance_dict['question']['choices']]
            }
    return all_dict
