import torch

from utils.layers import *
from modeling.text_encoder import TextEncoder,PromptTextEncoder, ClassifyMLPHead
from modeling.kg_encoder import RelationNet, Path_Encoder, PromptKGEncoder, SoftPromptEncoder, GPTGenerater
from transformers import GPT2Tokenizer

class LMRelationNet(nn.Module):
    def __init__(self, model_name, from_checkpoint,
                 concept_num, concept_dim, relation_num, relation_dim, concept_in_dim, hidden_size, num_hidden_layers,
                 num_attention_heads, fc_size, num_fc_layers, dropout, pretrained_concept_emb=None,
                 pretrained_relation_emb=None, freeze_ent_emb=True, init_range=0, ablation=None,
                 use_contextualized=False, emb_scale=1.0, encoder_config={}):
        super().__init__()
        self.use_contextualized = use_contextualized
        self.encoder = TextEncoder(model_name, from_checkpoint=from_checkpoint, **encoder_config)
        self.decoder = RelationNet(concept_num, concept_dim, relation_num, relation_dim, self.encoder.sent_dim, concept_in_dim,
                                   hidden_size, num_hidden_layers, num_attention_heads,
                                   fc_size, num_fc_layers, dropout, pretrained_concept_emb, pretrained_relation_emb,
                                   freeze_ent_emb=freeze_ent_emb, init_range=init_range, ablation=ablation,
                                   use_contextualized=use_contextualized, emb_scale=emb_scale)
        self.path_encoder = Path_Encoder(self.encoder.sent_dim)


    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension
        if self.use_contextualized:
            *lm_inputs, path_embedding, qa_ids, rel_ids, num_tuples, emb_data = inputs
        else:
            *lm_inputs, path_embedding, qa_ids, rel_ids, num_tuples = inputs
            emb_data = None
        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        agg_path_embedding = self.path_encoder(s=sent_vecs, p=path_embedding)
        logits, attn = self.decoder(path_embedding=agg_path_embedding, sent_vecs=sent_vecs, qa_ids=qa_ids, rel_ids=rel_ids, num_tuples=num_tuples, emb_data=emb_data)  # cxy-style param passing
        logits = logits.view(bs, nc)
        return logits, attn

class PromptLMRelationNet(nn.Module):
    def __init__(self, args, model_name, label_list_len, from_checkpoint,
                 concept_num, concept_dim, relation_num, relation_dim, concept_in_dim, hidden_size, num_hidden_layers,
                 prompt_token_num, fc_size, num_fc_layers, dropout, pretrained_concept_emb=None,
                 pretrained_relation_emb=None, freeze_ent_emb=True, init_range=0, ablation=None,
                 use_contextualized=False, emb_scale=1.0, encoder_config={}):
        super().__init__()
        self.args = args
        self.model_name = model_name
        self.use_contextualized = use_contextualized
        self.label_list = [0, 1]
        self.verbalize = {0: ["No"], 1: ["Yes"]}
        self.max_num_verbalizers = 1
        self.encoder = PromptTextEncoder(args, model_name, label_list_len, from_checkpoint=from_checkpoint)


        self.prompt_token_num = prompt_token_num
        self.kg_enc_out_size = self.encoder.sent_dim * prompt_token_num

        if self.args.input_format in ['soft_prompt_p_tuning', 'manual_hard_prompt']:
            self.decoder = SoftPromptEncoder(args=self.args, init_range=init_range, embed_size=self.encoder.sent_dim)
        elif self.args.input_format == 'GPT_kg_generator_as_prompt': # using gpt generate knowledge with input of "tail <sep> head"
            self.decoder = GPTGenerater(args=self.args, text_emb_size=self.encoder.sent_dim, init_range=init_range)
        else: # encoding relations
            self.decoder = PromptKGEncoder(concept_num, concept_dim, relation_num, relation_dim, self.encoder.sent_dim, concept_in_dim,
                                       hidden_size, num_hidden_layers, self.kg_enc_out_size,
                                       fc_size, num_fc_layers, dropout, pretrained_concept_emb, pretrained_relation_emb,
                                       freeze_ent_emb=freeze_ent_emb, init_range=init_range, ablation=ablation,
                                       use_contextualized=use_contextualized, emb_scale=emb_scale)



        self.m2c = self._build_mlm_logits_to_cls_logits_tensor()
        # self.path_encoder = Path_Encoder(self.encoder.sent_dim)


    def forward(self, *inputs, prompt_data, sample_ids=None, type=None):
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension
        if self.args.input_format in ['soft_prompt_p_tuning', 'manual_hard_prompt']:
            lm_inputs = inputs
        else:
            *lm_inputs, path_embedding, qa_ids, rel_ids, num_tuples = inputs

        input_ids, input_mask, segment_ids, output_mask = lm_inputs

        prompt_data = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in prompt_data]
        block_flag, mlm_mask, mlm_label = prompt_data # (bs*5, max_seq_len) (bs*5, 1)

        if self.args.input_format == 'manual_hard_prompt':
            outputs = self.encoder(input_ids=input_ids,
                                   attention_mask=input_mask,
                                   token_type_ids=None,
                                   labels=None)
            prediction_scores = outputs[0]  # (bs*5, max_len, vocab_size)
            masked_logits = prediction_scores[mlm_mask == 1]  # (bs*5, vocab_size)
            # (bs*5, 2)
            cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
            # (bs*5)
            mlm_label = mlm_label.view((-1))
            return cls_logits, mlm_label
        elif self.args.input_format == 'pg_kg_enc_as_prompt':
            # (bs*5, max_len)
            raw_embeds = self.encoder.module.roberta.embeddings.word_embeddings(input_ids)
            # (bs, num_path, hid_dim) -> (bs, hid_dim)
            agg_path_embedding = torch.mean(path_embedding, dim=1)
            # (bs, num_prompt*embed_size)
            replace_embeds = self.decoder(path_embedding=agg_path_embedding, sent_vecs=None, qa_ids=qa_ids, rel_ids=rel_ids, num_tuples=num_tuples, emb_data=None)  # cxy-style param passing
            bs = replace_embeds.shape[0]
            replace_embeds = replace_embeds.view(bs, self.prompt_token_num, -1)
            if replace_embeds.shape[-1] != raw_embeds.shape[-1]:
                print("the dim of soft prompt {} and raw embeddings {} is different".format(replace_embeds.shape[-1], raw_embeds.shape[-1]))
            blocked_indices = (block_flag == 1).nonzero().reshape((-1, self.prompt_token_num, 2))[:, :, 1]


            for bidx in range(bs):
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[bidx, i, :]

            inputs = {'inputs_embeds': raw_embeds, 'attention_mask': input_mask}

            outputs = self.encoder(inputs_embeds=inputs['inputs_embeds'],
                                    attention_mask=inputs['attention_mask'],
                                    token_type_ids=None)

            prediction_scores = outputs[0] # (bs*5, max_len, vocab_size)
            masked_logits = prediction_scores[mlm_mask == 1]  # (bs*5, vocab_size)
            # (bs*5, 2)
            cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
            # (bs*5)
            mlm_label = mlm_label.view((-1))
            return cls_logits, mlm_label
        elif self.args.input_format == 'GPT_kg_generator_as_prompt':
            # (bs*5, max_len)
            raw_embeds = self.encoder.module.roberta.embeddings.word_embeddings(input_ids)

            # (bs, num_prompt(num_context), embed_size)
            replace_embeds = self.decoder(type, sample_ids)
            bs = replace_embeds.shape[0]
            if replace_embeds.shape[-1] != raw_embeds.shape[-1]:
                print("the dim of soft prompt {} and raw embeddings {} is different".format(replace_embeds.shape[-1],
                                                                                            raw_embeds.shape[-1]))
            blocked_indices = (block_flag == 1).nonzero().reshape((-1, self.prompt_token_num, 2))[:, :, 1]

            for bidx in range(bs):
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[bidx, i, :]

            inputs = {'inputs_embeds': raw_embeds, 'attention_mask': input_mask}

            outputs = self.encoder(inputs_embeds=inputs['inputs_embeds'],
                                   attention_mask=inputs['attention_mask'],
                                   token_type_ids=None)

            prediction_scores = outputs[0]  # (bs*5, max_len, vocab_size)
            masked_logits = prediction_scores[mlm_mask == 1]  # (bs*5, vocab_size)
            # (bs*5, 2)
            cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
            # (bs*5)
            mlm_label = mlm_label.view((-1))
            return cls_logits, mlm_label
        elif self.args.input_format == 'soft_prompt_p_tuning':
            # (bs*5, max_len)
            if "roberta" in self.model_name:
                raw_embeds = self.encoder.module.roberta.embeddings.word_embeddings(input_ids)
            elif "albert" in self.model_name:
                raw_embeds = self.encoder.module.albert.embeddings.word_embeddings(input_ids)
            elif "gpt" in self.model_name:
                raw_embeds = self.encoder.module.transformer.wte(input_ids)
            bs = raw_embeds.shape[0]

            # (num_prompt, embed_size)
            device = raw_embeds.device
            replace_embeds = self.decoder(device)
            if replace_embeds.shape[-1] != raw_embeds.shape[-1]:
                print("the dim of soft prompt {} and raw embeddings {} is different".format(replace_embeds.shape[-1],
                                                                                            raw_embeds.shape[-1]))

            if (block_flag == 1).nonzero().shape[0] != 0:
                blocked_indices = (block_flag == 1).nonzero().reshape((-1, self.prompt_token_num, 2))[:, :, 1]
                for bidx in range(bs):
                    for i in range(blocked_indices.shape[1]):
                        # print(raw_embeds.shape, replace_embeds.shape)
                        raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
            # else:
            #     print("No prompt used!")


            inputs = {'inputs_embeds': raw_embeds, 'attention_mask': input_mask}

            if "albert" in self.model_name:
                outputs = self.encoder(inputs_embeds=inputs['inputs_embeds'],
                                       attention_mask=inputs['attention_mask'],
                                       token_type_ids=segment_ids)
            else:
                outputs = self.encoder(inputs_embeds=inputs['inputs_embeds'],
                                       attention_mask=inputs['attention_mask'],
                                       token_type_ids=None)

            prediction_scores = outputs[0]  # (bs*5, max_len, vocab_size)
            masked_logits = prediction_scores[mlm_mask == 1]  # (bs*5, vocab_size)
            # (bs*5, 2)
            cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
            # (bs*5)
            mlm_label = mlm_label.view((-1))
            return cls_logits, mlm_label



    def _convert_single_mlm_logits_to_cls_logits(self, logits):
        m2c = self.m2c.to(logits.device) # (2,1)

        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize[label]) for label in self.label_list], dtype=torch.float)
        filler_len = filler_len.to(logits.device) # (2)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)] # (2,1)
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len # (2)
        return cls_logits

    def _build_mlm_logits_to_cls_logits_tensor(self):
        # mrc: (2,1)
        m2c_tensor = torch.ones([len(self.label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(self.label_list):
            verbalizers = self.verbalize[label]
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.get_verbalization_ids(verbalizer, self.encoder.tokenizer, force_single_token=True)
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def get_verbalization_ids(self, word, tokenizer, force_single_token):
        """
        Get the token ids corresponding to a verbalization
        :param word: the verbalization
        :param tokenizer: the tokenizer to use
        :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
               If set to true, this method returns a single int instead of a list and throws an error if the word
               corresponds to multiple tokens.
        :return: either the list of token ids or the single token id corresponding to this word
        """
        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
        ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
        if not force_single_token:
            return ids
        assert len(ids) == 1, \
            f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
        verbalization_id = ids[0]
        assert verbalization_id not in tokenizer.all_special_ids, \
            f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
        assert verbalization_id != tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
        return verbalization_id

class PromptWithClassifyLMRelationNet(nn.Module):
    def __init__(self, args, model_name, label_list_len, from_checkpoint,
                 concept_num, concept_dim, relation_num, relation_dim, concept_in_dim, hidden_size, num_hidden_layers,
                 prompt_token_num, fc_size, num_fc_layers, dropout, pretrained_concept_emb=None,
                 pretrained_relation_emb=None, freeze_ent_emb=True, init_range=0, ablation=None,
                 use_contextualized=False, emb_scale=1.0, encoder_config={}):
        super().__init__()
        self.args = args
        self.model_name = model_name
        self.use_contextualized = use_contextualized
        self.label_list = [0, 1]
        self.verbalize = {0: ["No"], 1: ["Yes"]}
        self.max_num_verbalizers = 1
        self.encoder = PromptTextEncoder(args, model_name, label_list_len, from_checkpoint=from_checkpoint)


        self.prompt_token_num = prompt_token_num
        self.kg_enc_out_size = self.encoder.sent_dim * prompt_token_num

        self.decoder = SoftPromptEncoder(args=self.args, init_range=init_range, embed_size=self.encoder.sent_dim)


        self.classify_head = ClassifyMLPHead(input_size=self.encoder.hidden_dim, output_size=1, init_range=init_range)


    def forward(self, *inputs, prompt_data, sample_ids=None, type=None):
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension

        input_ids, input_mask, segment_ids, output_mask = inputs

        prompt_data = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in prompt_data]
        block_flag, mlm_mask, mlm_label = prompt_data # (bs*5, max_seq_len) (bs*5, 1)

        if "roberta" in self.model_name:
            raw_embeds = self.encoder.module.embeddings.word_embeddings(input_ids)
        elif "gpt" in self.model_name:
            raw_embeds = self.encoder.module.wte(input_ids)
        elif "albert" in self.model_name:
            raw_embeds = self.encoder.module.embeddings.word_embeddings(input_ids)

        bs = raw_embeds.shape[0]

        # (num_prompt, embed_size)
        device = raw_embeds.device
        replace_embeds = self.decoder(device)
        if replace_embeds.shape[-1] != raw_embeds.shape[-1]:
            print("the dim of soft prompt {} and raw embeddings {} is different".format(replace_embeds.shape[-1],
                                                                                        raw_embeds.shape[-1]))
        if (block_flag == 1).nonzero().shape[0] != 0:
            blocked_indices = (block_flag == 1).nonzero().reshape((-1, self.prompt_token_num, 2))[:, :, 1]
            for bidx in range(bs):
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        inputs = {'inputs_embeds': raw_embeds, 'attention_mask': input_mask}

        if "albert" in self.model_name:
            outputs = self.encoder(inputs_embeds=inputs['inputs_embeds'],
                                   attention_mask=inputs['attention_mask'],
                                   token_type_ids=segment_ids)
        else:
            outputs = self.encoder(inputs_embeds=inputs['inputs_embeds'],
                                   attention_mask=inputs['attention_mask'],
                                   token_type_ids=None)

        hidden_states = outputs[0]  # (bs*5, max_len, hid_dim)
        masked_hidden_state = hidden_states[mlm_mask == 1]  # (bs*5, hid_dim)
        cls_logits = self.classify_head(masked_hidden_state) # (bs*5, 1)
        cls_logits = cls_logits.view(-1, 5)
        return cls_logits, None

class PromptWithGenerateLMRelationNet(nn.Module):
    def __init__(self, args, model_name, label_list_len, from_checkpoint,
                 concept_num, concept_dim, relation_num, relation_dim, concept_in_dim, hidden_size, num_hidden_layers,
                 prompt_token_num, fc_size, num_fc_layers, dropout, pretrained_concept_emb=None,
                 pretrained_relation_emb=None, freeze_ent_emb=True, init_range=0, ablation=None,
                 use_contextualized=False, emb_scale=1.0, encoder_config={}):
        super().__init__()
        self.args = args
        self.model_name = model_name
        self.use_contextualized = use_contextualized
        self.label_list = [0, 1]
        self.verbalize = {0: ["No"], 1: ["Yes"]}
        self.max_num_verbalizers = 1
        self.encoder = PromptTextEncoder(model_name, label_list_len, from_checkpoint=from_checkpoint)

        self.prompt_token_num = prompt_token_num
        self.kg_enc_out_size = self.encoder.sent_dim * prompt_token_num

        self.decoder = SoftPromptEncoder(args=self.args, init_range=init_range, embed_size=self.encoder.sent_dim)
        self.m2c = self._build_mlm_logits_to_cls_logits_tensor()

    def forward(self, *inputs, prompt_data, sample_ids=None, type=None):
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in
                  inputs]  # merge the batch dimension and the num_choice dimension

        input_ids, input_mask, segment_ids, output_mask = inputs


        prompt_data = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in prompt_data]
        block_flag, mlm_mask, mlm_label = prompt_data  # (bs*5, max_seq_len) (bs*5, 1)

        with torch.no_grad():
            if "roberat" in self.model_name:
                raw_embeds = self.encoder.module.roberta.embeddings.word_embeddings(input_ids)
            elif "gpt" in self.model_name:
                # raw_embeds = self.encoder.module.wte(input_ids)
                raw_embeds = self.encoder.module.transformer.wte(input_ids)
        bs = raw_embeds.shape[0]

        # (num_prompt, embed_size)
        device = raw_embeds.device
        replace_embeds = self.decoder(device)
        if replace_embeds.shape[-1] != raw_embeds.shape[-1]:
            print("the dim of soft prompt {} and raw embeddings {} is different".format(replace_embeds.shape[-1],
                                                                                        raw_embeds.shape[-1]))

        blocked_indices = (block_flag == 1).nonzero().reshape((-1, self.prompt_token_num, 2))[:, :, 1]

        for bidx in range(bs):
            for i in range(blocked_indices.shape[1]):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        inputs = {'inputs_embeds': raw_embeds, 'attention_mask': input_mask}

        outputs = self.encoder(inputs_embeds=inputs['inputs_embeds'],
                               attention_mask=inputs['attention_mask'],
                               token_type_ids=None)

        pred_logits = outputs[0]  # (bs*5, max_len, vocab_size)
        y_pred_logits = pred_logits[:, :-1, :]
        y_pred_logits = y_pred_logits.contiguous().view(-1, pred_logits.shape[2]) # (bs*5*max_len, vocab_size)
        y_target_ids = input_ids[:, 1:]
        y_target_ids = y_target_ids.contiguous().view(-1) # (vocab_size)
        assert y_target_ids.shape[0] == y_pred_logits.shape[0]
        # label_mask = torch.zeros_like(input_mask)
        # label_mask[block_flag == 1] = 1
        # label_mask = label_mask[:, :-1] # (bs*5, max_len-1)
        # label_mask = label_mask.contiguous().view(-1, 5, pred_logits.shape[1]-1) # (bs, 5, max_len-1)
        label_mask = mlm_mask[:, 1:]
        label_mask = label_mask.contiguous().view(-1, 5, pred_logits.shape[1] - 1)  # (bs, 5, max_len-1)

        prediction_scores = outputs[0]  # (bs*5, max_len, vocab_size)
        masked_logits = prediction_scores[mlm_mask == 1]  # (bs*5, vocab_size)
        # (bs*5, 2)
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        # (bs*5)
        mlm_label = mlm_label.view((-1))
        eval_logits = cls_logits
        return y_pred_logits, y_target_ids, label_mask, eval_logits


    def _convert_single_mlm_logits_to_cls_logits(self, logits):
        m2c = self.m2c.to(logits.device) # (2,1)

        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize[label]) for label in self.label_list], dtype=torch.float)
        filler_len = filler_len.to(logits.device) # (2)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)] # (2,1)
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len # (2)
        return cls_logits

    def _build_mlm_logits_to_cls_logits_tensor(self):
        # mrc: (2,1)
        m2c_tensor = torch.ones([len(self.label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(self.label_list):
            verbalizers = self.verbalize[label]
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.get_verbalization_ids(verbalizer, self.encoder.tokenizer, force_single_token=True)
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def get_verbalization_ids(self, word, tokenizer, force_single_token):
        """
        Get the token ids corresponding to a verbalization
        :param word: the verbalization
        :param tokenizer: the tokenizer to use
        :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
               If set to true, this method returns a single int instead of a list and throws an error if the word
               corresponds to multiple tokens.
        :return: either the list of token ids or the single token id corresponding to this word
        """
        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
        ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
        if not force_single_token:
            return ids
        assert len(ids) == 1, \
            f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
        verbalization_id = ids[0]
        assert verbalization_id not in tokenizer.all_special_ids, \
            f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
        assert verbalization_id != tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
        return verbalization_id