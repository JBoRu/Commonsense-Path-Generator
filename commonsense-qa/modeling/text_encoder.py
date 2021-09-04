from transformers import *
from utils.layers import *
from utils.data_utils import get_gpt_token_num, GPT_SPECIAL_TOKENS

MODEL_CLASS_TO_NAME = {
    # 'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'gpt': list(GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'albert': list(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}

MODEL_CLASSES_PROMPT_MLM = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model': BertForMaskedLM
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model': RobertaForMaskedLM
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertForMaskedLM
    },
    'gpt': {
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel
    },
}
MODEL_CLASSES_PROMPT_BASE = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model': BertModel
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model': RobertaModel
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertModel
    },
    'gpt': {
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2Model
    }
}

class LSTMTextEncoder(nn.Module):
    pool_layer_classes = {'mean': MeanPoolLayer, 'max': MaxPoolLayer}

    def __init__(self, vocab_size=1, emb_size=300, hidden_size=300, output_size=300, num_layers=2, bidirectional=True,
                 emb_p=0.0, input_p=0.0, hidden_p=0.0, pretrained_emb_or_path=None, freeze_emb=True,
                 pool_function='max', output_hidden_states=False):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.output_hidden_states = output_hidden_states
        assert not bidirectional or hidden_size % 2 == 0

        if pretrained_emb_or_path is not None:
            if isinstance(pretrained_emb_or_path, str):  # load pretrained embedding from a .npy file
                pretrained_emb_or_path = torch.tensor(np.load(pretrained_emb_or_path), dtype=torch.float)
            emb = nn.Embedding.from_pretrained(pretrained_emb_or_path, freeze=freeze_emb)
            emb_size = emb.weight.size(1)
        else:
            emb = nn.Embedding(vocab_size, emb_size)
        self.emb = EmbeddingDropout(emb, emb_p)
        self.rnns = nn.ModuleList([nn.LSTM(emb_size if l == 0 else hidden_size,
                                           (hidden_size if l != num_layers else output_size) // (2 if bidirectional else 1),
                                           1, bidirectional=bidirectional, batch_first=True) for l in range(num_layers)])
        self.pooler = self.pool_layer_classes[pool_function]()

        self.input_dropout = nn.Dropout(input_p)
        self.hidden_dropout = nn.ModuleList([RNNDropout(hidden_p) for _ in range(num_layers)])

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)

        returns: tensor of shape (batch_size, hidden_size)
        """
        assert (lengths > 0).all()
        batch_size, seq_len = inputs.size()
        hidden_states = self.input_dropout(self.emb(inputs))
        all_hidden_states = [hidden_states]
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dropout)):
            hidden_states = pack_padded_sequence(hidden_states, lengths, batch_first=True, enforce_sorted=False)
            hidden_states, _ = rnn(hidden_states)
            hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True, total_length=seq_len)
            all_hidden_states.append(hidden_states)
            if l != self.num_layers - 1:
                hidden_states = hid_dp(hidden_states)
        pooled = self.pooler(all_hidden_states[-1], lengths)
        assert len(all_hidden_states) == self.num_layers + 1
        outputs = (all_hidden_states[-1], pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        return outputs

class TextEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, output_token_states=False, from_checkpoint=None, **kwargs):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.output_token_states = output_token_states
        assert not self.output_token_states or self.model_type in ('bert', 'roberta', 'albert')

        if self.model_type in ('lstm',):
            self.module = LSTMTextEncoder(**kwargs, output_hidden_states=True)
            self.sent_dim = self.module.output_size
        else:
            module_config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, cache_dir='/mnt/nlp_model/huggingface')
            self.module = AutoModel.from_pretrained(model_name, config=module_config, cache_dir='/mnt/nlp_model/huggingface')
            if not from_checkpoint == 'None':
                # self.module = self.module.from_pretrained(from_checkpoint, config=module_config, cache_dir='../cache/')
                weight = torch.load(from_checkpoint, map_location='cpu')
                new_dict = {}
                for k, v in weight.items():
                    nk = k.replace('_transformer_model.', '')
                    if nk not in self.module.state_dict():
                        print(k)
                        continue
                    new_dict[nk] = v
                model_dict = self.module.state_dict()
                model_dict.update(new_dict)
                self.module.load_state_dict(model_dict)

            if self.model_type in ('gpt',):
                self.module.resize_token_embeddings(get_gpt_token_num())

            self.sent_dim = self.module.config.n_embd if self.model_type in ('gpt',) else self.module.config.hidden_size
        print(self.model_type)

    def forward(self, *inputs, layer_id=-1):
        '''
        layer_id: only works for non-LSTM encoders
        output_token_states: if True, return hidden states of specific layer and attention masks
        '''

        if self.model_type in ('lstm',):  # lstm
            input_ids, lengths = inputs
            outputs = self.module(input_ids, lengths)
        elif self.model_type in ('gpt',):  # gpt
            input_ids, cls_token_ids, lm_labels = inputs  # lm_labels is not used
            outputs = self.module(input_ids)
        else:  # bert / xlnet / roberta
            input_ids, attention_mask, token_type_ids, output_mask = inputs
            outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]

        if self.model_type in ('lstm',):
            sent_vecs = outputs[1]
        elif self.model_type in ('gpt',):
            cls_token_ids = cls_token_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
            sent_vecs = hidden_states.gather(1, cls_token_ids).squeeze(1)
        elif self.model_type in ('xlnet',):
            sent_vecs = hidden_states[:, -1]
        else:
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = hidden_states[:, 0]
        return sent_vecs, all_hidden_states

class PromptTextEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, args, model_name, label_list_len, from_checkpoint=None):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        if 'classify' in args.input_format:
            self.model_dict = MODEL_CLASSES_PROMPT_BASE[self.model_type]
        else:
            self.model_dict = MODEL_CLASSES_PROMPT_MLM[self.model_type]
        print(self.model_dict)
        # self.model_dict = MODEL_CLASSES_PROMPT[self.model_type]
        # if args.input_format == "p-tuning-GPT-classify":
        #     print("Using GPT2Model for mlp classify")
        #     self.model_dict['model'] = GPT2Model

        print("Loading plm config....")
        config_class = self.model_dict['config']
        if self.model_type == 'roberta':
            path = "/mnt/nlp_model/huggingface/roberta-large/"
        elif self.model_type == 'gpt':
            if model_name == 'gpt2':
                path = "/mnt/nlp_model/huggingface/gpt2/"
            elif model_name == 'gpt2-medium':
                path = "/mnt/nlp_model/gpt2-medium/"
        elif self.model_type == 'albert':
            path = "/mnt/nlp_model/albert-xxlarge-v2/"

        model_config = config_class.from_pretrained(path, num_labels=label_list_len)

        print("Loading plm tokenizer....")
        tokenizer_class = self.model_dict['tokenizer']
        self.tokenizer = tokenizer_class.from_pretrained(path)

        print("Loading plm model....")
        model_class = self.model_dict['model']
        self.module = model_class.from_pretrained(path, config=model_config)

        if not from_checkpoint == 'None':
            # self.module = self.module.from_pretrained(from_checkpoint, config=module_config, cache_dir='../cache/')
            weight = torch.load(from_checkpoint, map_location='cpu')
            new_dict = {}
            for k, v in weight.items():
                nk = k.replace('_transformer_model.', '')
                if nk not in self.module.state_dict():
                    print(k)
                    continue
                new_dict[nk] = v
            model_dict = self.module.state_dict()
            model_dict.update(new_dict)
            self.module.load_state_dict(model_dict)

        if self.model_type in ('gpt',):
            self.tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
            self.module.resize_token_embeddings(len(self.tokenizer))

        if self.model_type in ('gpt', ):
            self.sent_dim = self.module.config.n_embd
        elif self.model_type in ('albert', ):
            self.sent_dim = self.module.config.embedding_size
        else:
            self.sent_dim = self.module.config.hidden_size

        if self.model_type in ('albert',):
            self.hidden_dim = self.module.config.hidden_size
        else:
            self.hidden_dim = self.sent_dim

        print(self.model_type)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None):
        if input_ids is not None:
            return self.module(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        else:
            return self.module(inputs_embeds=inputs_embeds,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

class ClassifyMLPHead(nn.Module):
    def __init__(self, input_size, output_size, init_range):
        super(ClassifyMLPHead, self).__init__()
        self.init_range = init_range
        self.classify_head = nn.Linear(input_size, output_size, bias=False)

        if self.init_range > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input):
        output = self.classify_head(input) # (bs, 1)
        return output