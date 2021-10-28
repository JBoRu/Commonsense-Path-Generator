import random

import torch

from utils.data_utils import *
from modeling.text_encoder import MODEL_NAME_TO_CLASS

class LMRelationNetDataLoader(object):

    def __init__(self, args, train_statement_path,
                 dev_statement_path,
                 test_statement_path,
                 batch_size, eval_batch_size, device, model_name,
                 max_seq_length=128,
                 is_inhouse=True, inhouse_train_qids_path=None):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse
        self.use_contextualized = False

        model_type = MODEL_NAME_TO_CLASS[model_name]

        # (num_example), (num_example, 5, max_seq_lem), (num_example, 5)
        print("Load and process input data")
        self.train_qids, self.train_labels, *self.train_data, self.prompt_train_data = load_input_tensors(self.args, train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_labels, *self.dev_data, self.prompt_dev_data = load_input_tensors(self.args, dev_statement_path, model_type, model_name, max_seq_length)

        num_choice = self.train_data[0].size(1)

        # # (num_samples, 5, 5, hid_dim)
        # print("Load path embedding data")
        # with open(path_embedding_path, 'rb') as handle:
        #     path_embedding = pickle.load(handle)
        # self.train_data += [path_embedding['train']]
        # self.dev_data += [path_embedding['dev']]
        #
        # # 5
        # print("Load 2hop path data")
        # self.train_data += load_2hop_relational_paths(train_rpath_jsonl, train_adj_path,
        #                                               emb_pk_path=train_node_features_path if use_contextualized else None,
        #                                               max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)
        # self.dev_data += load_2hop_relational_paths(dev_rpath_jsonl, dev_adj_path,
        #                                             emb_pk_path=dev_node_features_path if use_contextualized else None,
        #                                             max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)
        #
        # assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
        # assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_data)
        # if test_statement_path is not None:
        #     self.test_qids, self.test_labels, *self.test_data, self.prompt_test_data = load_input_tensors(self.args, test_statement_path, model_type, model_name, max_seq_length)
        #     self.test_data += [path_embedding['test']]
        #     self.test_data += load_2hop_relational_paths(test_rpath_jsonl, test_adj_path,
        #                                                  emb_pk_path=test_node_features_path if use_contextualized else None,
        #                                                  max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)
        #     assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_data)
        # num_tuple_idx = -2 if use_contextualized else -1
        # print('| train_num_tuples = {:.2f} | dev_num_tuples = {:.2f} | test_num_tuples = {:.2f} |'.format(self.train_data[num_tuple_idx].float().mean(),
        #                                                                                                   self.dev_data[num_tuple_idx].float().mean(),
        #                                                                                                   self.test_data[num_tuple_idx].float().mean() if test_statement_path else 0))

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

    def __getitem__(self, index):
        raise NotImplementedError()

    def get_node_feature_dim(self):
        return self.train_data[-1].size(-1) if self.use_contextualized else None

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return BatchGenerator(self.device, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors=self.train_data, prompt_data=self.prompt_train_data)

    def train_eval(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors=self.train_data, prompt_data=self.prompt_train_data)

    def dev(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors=self.dev_data, prompt_data=self.prompt_dev_data)

    def test(self):
        if self.is_inhouse:
            return BatchGenerator(self.device, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors=self.train_data, prompt_data=self.prompt_train_data)
        else:
            return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors=self.test_data, prompt_data=self.prompt_test_data)

class KCRDataLoader(object):

    def __init__(self, args, train_statement_path,
                 dev_statement_path,
                 test_statement_path,
                 batch_size, eval_batch_size, device, model_name,
                 max_seq_length=128,
                 is_inhouse=True, inhouse_train_qids_path=None):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]
        if args.dataset == 'csqa':
            self.load_data_from_kcr(train_statement_path,dev_statement_path, inhouse_train_qids_path,
                                    model_type, model_name, max_seq_length)
        elif args.dataset == 'csqav2':
            self.load_data_from_csqav2(train_statement_path, dev_statement_path, inhouse_train_qids_path,
                                    model_type, model_name, max_seq_length)


    def load_data_from_kcr(self,train_statement_path,dev_statement_path,inhouse_train_qids_path,
                           model_type,model_name,max_seq_length):
        # (num_example), (num_example, 5, max_seq_lem), (num_example, 5)
        print("Load and process input data")
        self.train_qids, self.train_labels, *self.train_data, self.prompt_train_data = \
            load_input_tensors_for_kcr(self.args, train_statement_path, model_type, model_name, max_seq_length, "train")
        self.dev_qids, self.dev_labels, *self.dev_data, self.prompt_dev_data = \
            load_input_tensors_for_kcr(self.args, dev_statement_path, model_type, model_name, max_seq_length, "eval")

        # num_choice = self.train_data[0].size(1)

    def load_data_from_csqav2(self,train_statement_path,dev_statement_path,inhouse_train_qids_path,
                           model_type,model_name,max_seq_length):
        # (num_example), (num_example, 5, max_seq_lem), (num_example, 5)
        print("Load and process input data")
        self.train_qids, self.train_labels, *self.train_data, self.prompt_train_data = \
            load_input_tensors_for_csqav2(self.args, train_statement_path, model_type, model_name, max_seq_length, "train")
        self.dev_qids, self.dev_labels, *self.dev_data, self.prompt_dev_data = \
            load_input_tensors_for_csqav2(self.args, dev_statement_path, model_type, model_name, max_seq_length, "eval")

        # num_choice = self.train_data[0].size(1)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)

            self.pseudo_test_qids, self.pseudo_test_labels, *self.pseudo_train_data, self.pseudo_prompt_train_data = \
                load_input_tensors_for_csqav2(self.args, train_statement_path, model_type, model_name,
                                           max_seq_length, "eval")

            self.inhouse_train_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])


    def __getitem__(self, index):
        raise NotImplementedError()

    def get_node_feature_dim(self):
        return None

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return BatchGenerator(self.device, self.batch_size, train_indexes,
                              self.train_qids, self.train_labels, tensors=self.train_data,
                              prompt_data=self.prompt_train_data)

    def train_eval(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors=self.train_data, prompt_data=self.prompt_train_data)

    def dev(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors=self.dev_data, prompt_data=self.prompt_dev_data)

    def test(self):
        if self.is_inhouse:
            return BatchGenerator(self.device, self.eval_batch_size, self.inhouse_test_indexes,
                                  self.pseudo_test_qids, self.pseudo_test_labels, tensors=self.pseudo_train_data,
                                  prompt_data=self.pseudo_prompt_train_data)
        else:
            return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)),
                                  self.test_qids, self.test_labels, tensors=self.test_data,
                                  prompt_data=self.prompt_test_data)

class NLIDataLoader(object):

    def __init__(self, args, train_statement_path, dev_statement_path, test_statement_path,
                 batch_size, eval_batch_size, device, model_name,
                 max_seq_length=128, num_label=3):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device

        model_class = MODEL_NAME_TO_CLASS[model_name]

        # (num_example), (num_example, 5, max_seq_lem), (num_example, 5)
        print("Load and process input data")
        if self.args.mode == "train":
            self.train_qids, self.train_labels, *self.train_data, self.prompt_train_data = \
                load_input_tensors_for_nli(self.args, train_statement_path, model_class, model_name, max_seq_length,
                                           "train", num_label, cache=self.args.cache)

        self.dev_qids, self.dev_labels, *self.dev_data, self.prompt_dev_data = \
            load_input_tensors_for_nli(self.args, dev_statement_path, model_class, model_name, max_seq_length,
                                       "eval", num_label, cache=self.args.cache)

        self.test_qids, self.test_labels, *self.test_data, self.prompt_test_data = \
            load_input_tensors_for_nli(self.args, test_statement_path, model_class, model_name, max_seq_length,
                                       "eval", num_label, cache=self.args.cache)

    def __getitem__(self, index):
        raise NotImplementedError()

    def train_size(self):
        return len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        return len(self.test_qids)

    def train(self):
        train_indexes = torch.randperm(len(self.train_qids))
        return BatchGenerator(self.device, self.batch_size, train_indexes, self.train_qids, self.train_labels,
                              tensors=self.train_data, prompt_data=self.prompt_train_data)

    def dev(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids,
                              self.dev_labels, tensors=self.dev_data, prompt_data=self.prompt_dev_data)

    def test(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids,
                              self.test_labels, tensors=self.test_data, prompt_data=self.prompt_test_data)

class MixedDataLoader(object):

    def __init__(self, args, train_statement_path, dev_statement_path, test_statement_path,
                 batch_size, eval_batch_size, device, model_name,
                 max_seq_length=128, num_label=3, is_inhouse=True, inhouse_train_qids_path=None):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse

        model_class = MODEL_NAME_TO_CLASS[model_name]

        # (num_example), (num_example, 5, max_seq_lem), (num_example, 5)
        print("Load and process input nli data")
        self.train_nli_qids, self.train_nli_labels, *self.train_nli_data, self.prompt_nli_train_data = \
            load_input_tensors_for_nli(self.args, args.train_nli_path, model_class, model_name, max_seq_length,
                                       "train", num_label, cache=self.args.cache)

        print("Load and process input csqa data")
        self.train_csqa_qids, self.train_csqa_labels, *self.train_csqa_data, self.prompt_csqa_train_data = \
            load_input_tensors_for_kcr(self.args, train_statement_path, model_class, model_name, max_seq_length, "train")

        self.dev_qids, self.dev_labels, *self.dev_data, self.prompt_dev_data = \
            load_input_tensors_for_kcr(self.args, dev_statement_path, model_class, model_name, max_seq_length, "eval")

        if self.is_inhouse:
            self.pseudo_test_qids, self.pseudo_test_labels, *self.pseudo_train_data, self.pseudo_prompt_train_data = \
                load_input_tensors_for_kcr(self.args, train_statement_path, model_class, model_name,
                                           max_seq_length, "eval")
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_csqa_indexes = [i for i, qid in enumerate(self.train_csqa_qids) if qid in inhouse_qids]
            self.inhouse_test_indexes = [i for i, qid in enumerate(self.train_csqa_qids) if qid not in inhouse_qids]

        print("Mixed csqa train data and nli train data")

        self.train_nli_indexs = [ i + len(self.train_csqa_qids) for i,_ in enumerate(self.train_nli_qids)]
        self.mixed_train_qids = self.train_csqa_qids + self.train_nli_qids
        self.mixed_train_indexs = self.inhouse_train_csqa_indexes + self.train_nli_indexs
        self.mixed_train_labels = torch.cat([self.train_csqa_labels, self.train_nli_labels],dim=0)
        self.mixed_train_data = [torch.cat([csqa, nli], dim=0) for csqa, nli in zip(self.train_csqa_data, self.train_nli_data)]
        self.mixed_prompt_train_data = [torch.cat([csqa, nli], dim=0) for csqa, nli in zip(self.prompt_csqa_train_data, self.prompt_nli_train_data)]



    def __getitem__(self, index):
        raise NotImplementedError()

    def train_size(self):
        return len(self.mixed_train_indexs)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        return len(self.pseudo_test_qids)

    def train(self):
        random.shuffle(self.mixed_train_indexs)
        mixed_train_indexs = torch.tensor(self.mixed_train_indexs)
        return BatchGenerator(self.device, self.batch_size, mixed_train_indexs,
                              self.mixed_train_qids, self.mixed_train_labels, tensors=self.mixed_train_data,
                              prompt_data=self.mixed_prompt_train_data)

    def dev(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids,
                              self.dev_labels, tensors=self.dev_data, prompt_data=self.prompt_dev_data)

    def test(self):
        if self.is_inhouse:
            inhouse_test_indexes = torch.tensor(self.inhouse_test_indexes)
            return BatchGenerator(self.device, self.eval_batch_size, inhouse_test_indexes,
                                  self.pseudo_test_qids, self.pseudo_test_labels, tensors=self.pseudo_train_data,
                                  prompt_data=self.pseudo_prompt_train_data)
        else:
            return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)),
                                  self.test_qids, self.test_labels, tensors=self.test_data,
                                  prompt_data=self.prompt_test_data)