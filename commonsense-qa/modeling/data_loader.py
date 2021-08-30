from utils.data_utils import *
from modeling.text_encoder import MODEL_NAME_TO_CLASS

class LMRelationNetDataLoader(object):

    def __init__(self, args, path_embedding_path, train_statement_path, train_rpath_jsonl,
                 dev_statement_path, dev_rpath_jsonl,
                 test_statement_path, test_rpath_jsonl,
                 batch_size, eval_batch_size, device, model_name,
                 max_tuple_num=200, max_seq_length=128,
                 is_inhouse=True, inhouse_train_qids_path=None, use_contextualized=False,
                 train_adj_path=None, train_node_features_path=None, dev_adj_path=None, dev_node_features_path=None,
                 test_adj_path=None, test_node_features_path=None, node_feature_type=None):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse
        self.use_contextualized = use_contextualized

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

class LMRelationNetDataLoaderForPred(object):

    def __init__(self, path_embedding_path,
                 test_statement_path, test_rpath_jsonl,
                 batch_size, eval_batch_size, device, model_name,
                 max_tuple_num=200, max_seq_length=128,
                 is_inhouse=True, inhouse_train_qids_path=None, use_contextualized=False,
                 test_adj_path=None, test_node_features_path=None, node_feature_type=None):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse
        self.use_contextualized = use_contextualized

        model_type = MODEL_NAME_TO_CLASS[model_name]

        num_choice = self.train_data[0].size(1)

        with open(path_embedding_path, 'rb') as handle:
            path_embedding = pickle.load(handle)

        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_data)
        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_data = load_input_tensors(self.args, test_statement_path, model_type, model_name, max_seq_length)
            self.test_data += [path_embedding['test']]
            self.test_data += load_2hop_relational_paths(test_rpath_jsonl, test_adj_path,
                                                         emb_pk_path=test_node_features_path if use_contextualized else None,
                                                         max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)
            assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_data)

        num_tuple_idx = -2 if use_contextualized else -1
        print('| train_num_tuples = {:.2f} | dev_num_tuples = {:.2f} | test_num_tuples = {:.2f} |'.format(self.train_data[num_tuple_idx].float().mean(),
                                                                                                          self.dev_data[num_tuple_idx].float().mean(),
                                                                                                          self.test_data[num_tuple_idx].float().mean() if test_statement_path else 0))

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
        return BatchGenerator(self.device, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors=self.train_data)

    def train_eval(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors=self.train_data)

    def dev(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors=self.dev_data)

    def test(self):
        if self.is_inhouse:
            return BatchGenerator(self.device, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors=self.train_data)
        else:
            return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors=self.test_data)
