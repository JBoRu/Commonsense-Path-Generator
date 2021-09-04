import os
from utils.layers import *
import transformers
# assert transformers.__version__ == '2.8.0'
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
# from transformers import *

from utils.data_helper import DataHelper
from modeling.generator import *
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

def init_weights_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.02)

class Path_Encoder(nn.Module):
    """docstring for Classifier"""
    def __init__(self, input_dim_bert, input_dim_gpt=768):
        super().__init__()

        self.input_dim_gpt = input_dim_gpt
        self.input_dim_bert = input_dim_bert

        self.attention = nn.Sequential( 
                            nn.Linear(self.input_dim_gpt, self.input_dim_bert),
                            nn.Tanh(),
                        )
        self.attention.apply(init_weights_normal)

    def forward(self, s, p):
        # choice: [batch, hidden]
        # context: [batch, context, hidden]

        batch_size, num_context, _ = p.size()

        # attention
        # q_T*W(p)
        query = s.view(batch_size, 1, self.input_dim_bert)
        alpha = (self.attention(p) * query).sum(-1, keepdim=True)
        alpha = F.softmax(alpha, dim=-2)
        context = (alpha * p).sum(-2)

        return context

class GPTGenerater(nn.Module):

    def __init__(self, args, text_emb_size, init_range):
        super(GPTGenerater, self).__init__()

        self.args = args
        self.init_range = init_range
        print("Load dataset of paths")
        self.datahelper = DataHelper(args)

        print("Init GPT generator")
        lm_path = '/mnt/nlp_model/gpt2/models/gpt2_small/'
        config = GPT2Config.from_pretrained(lm_path)
        gpt = GPT2Model.from_pretrained(lm_path)
        config.vocab_size = len(self.datahelper.gpt_tokenizer)
        gpt.resize_token_embeddings(len(self.datahelper.gpt_tokenizer))
        pretrain_generator_ckpt = os.path.join('./saved_models/pretrain_generator', 'commonsense-path-generator.ckpt')

        self.generator = GeneratorForPrompt(gpt, config, max_len=args.output_len)
        print("Load pretrained checkpoint!")
        self.generator.load_state_dict(torch.load(pretrain_generator_ckpt, map_location='cpu'))

        self.mlp = MLP(self.generator.hid_size, 2 * text_emb_size, text_emb_size, 1, dropout=0.1,
                       batch_norm=False, layer_norm=True)
        if self.init_range > 0:
            self.mlp.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, type, sample_ids):
        if type == 'train':
            context_embedding = self._get_path_embedding_greedy(self.datahelper.trainset.tensors[0], sample_ids, self.generator, self.args)
        elif type == 'dev':
            context_embedding = self._get_path_embedding_greedy(self.datahelper.devset.tensors[0], sample_ids, self.generator, self.args)
        elif type == 'test':
            if self.args.inhouse:
                context_embedding = self._get_path_embedding_greedy(self.datahelper.trainset.tensors[0], sample_ids, self.generator, self.args)
            else:
                context_embedding = self._get_path_embedding_greedy(self.datahelper.testset.tensors[0], sample_ids, self.generator, self.args)

        context_embedding = self.mlp(context_embedding) # (bs, num_context, emb_size)
        return context_embedding

    def _get_path_embedding_greedy(self, dataset, sample_ids, generator, args, tokenizer=None, output_file=None):

        dataset = dataset.to(sample_ids.device)
        batch_data = dataset[sample_ids]
        assert batch_data.shape[0] == sample_ids.shape[0], "batch data length not equal"

        context = batch_data
        # context = context[0].to(args.device)
        batch_size, num_choice, num_context, context_len = context.size()
        context = context.view(-1, context_len)
        context_embedding = self.generator(context) # (-1, hid)
        # (bs, num_choice, num_path, hid)

        context_embedding = context_embedding.view(batch_size*num_choice, num_context, -1)

        # if not output_file is None:
        #     for path in generated_paths:
        #         path = tokenizer.decode(path.tolist(), skip_special_tokens=True)
        #         path = ' '.join(path.replace('<PAD>', '').split())
        #         output_file.write(path + '\n')

        # path_embeddings.extend(context_embedding.tolist())

        return context_embedding # (bs, num_context, hid)

class RelationNet(nn.Module):

    def __init__(self, concept_num, concept_dim, relation_num, relation_dim, sent_dim, concept_in_dim,
                 hidden_size, num_hidden_layers, num_attention_heads, fc_size, num_fc_layers, dropout,
                 pretrained_concept_emb=None, pretrained_relation_emb=None, freeze_ent_emb=True,
                 init_range=0, ablation=None, use_contextualized=False, emb_scale=1.0, path_embedding_dim=768):

        super().__init__()
        self.init_range = init_range
        self.relation_num = relation_num
        self.ablation = ablation

        self.rel_emb = nn.Embedding(relation_num, relation_dim)
        self.concept_emb = CustomizedEmbedding(concept_num=concept_num, concept_out_dim=concept_dim,
                                               use_contextualized=use_contextualized, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                               scale=emb_scale)

        encoder_dim = {'no_qa': relation_dim, 'no_2hop_qa': relation_dim, 'no_rel': concept_dim * 2}.get(self.ablation, concept_dim * 2 + relation_dim)
        if self.ablation in ('encode_qas',):
            encoder_dim += sent_dim
        self.mlp = MLP(encoder_dim, hidden_size * 2, hidden_size,
                       num_hidden_layers, dropout, batch_norm=False, layer_norm=True)

        if ablation in ('multihead_pool',):
            self.attention = MultiheadAttPoolLayer(num_attention_heads, sent_dim, hidden_size)
        elif ablation in ('att_pool',):
            self.attention = AttPoolLayer(sent_dim, hidden_size)

        self.dropout_m = nn.Dropout(dropout)
        self.hid2out = MLP(path_embedding_dim + hidden_size + sent_dim, fc_size, 1, num_fc_layers, dropout, batch_norm=False, layer_norm=True)
        self.activation = GELU()

        if self.init_range > 0:
            self.apply(self._init_weights)

        if pretrained_relation_emb is not None and ablation not in ('randomrel',):
            self.rel_emb.weight.data.copy_(pretrained_relation_emb)

        if pretrained_concept_emb is not None and not use_contextualized:
            self.concept_emb.emb.weight.data.copy_(pretrained_concept_emb)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, path_embedding, sent_vecs, qa_ids, rel_ids, num_tuples, emb_data=None):
        """
        sent_vecs: tensor of shape (batch_size, d_sent)
        qa_ids: tensor of shape (batch_size, max_tuple_num, 2)
        rel_ids: tensor of shape (batch_size, max_tuple_num)
        num_tuples: tensor of shape (batch_size,)
        (emb_data: tensor of shape (batch_size, max_cpt_num, emb_dim))
        """

        bs, sl, _ = qa_ids.size()
        mask = torch.arange(sl, device=qa_ids.device) >= num_tuples.unsqueeze(1)
        if self.ablation in ('no_1hop', 'no_2hop', 'no_2hop_qa'):
            n_1hop_rel = int(np.sqrt(self.relation_num))
            assert n_1hop_rel * (n_1hop_rel + 1) == self.relation_num
            valid_mask = rel_ids > n_1hop_rel if self.ablation == 'no_1hop' else rel_ids <= n_1hop_rel
            mask = mask | ~valid_mask
        mask[mask.all(1), 0] = 0  # a temporary solution for instances that have no qar-pairs

        qa_emb = self.concept_emb(qa_ids.view(bs, -1), emb_data).view(bs, sl, -1)
        rel_embed = self.rel_emb(rel_ids)

        if self.ablation not in ('no_factor_mul',):
            n_1hop_rel = int(np.sqrt(self.relation_num))
            assert n_1hop_rel * (n_1hop_rel + 1) == self.relation_num
            rel_ids = rel_ids.view(bs * sl)
            twohop_mask = rel_ids >= n_1hop_rel
            twohop_rel = rel_ids[twohop_mask] - n_1hop_rel
            r1, r2 = twohop_rel // n_1hop_rel, twohop_rel % n_1hop_rel
            assert (r1 >= 0).all() and (r2 >= 0).all() and (r1 < n_1hop_rel).all() and (r2 < n_1hop_rel).all()
            rel_embed = rel_embed.view(bs * sl, -1)
            rel_embed[twohop_mask] = torch.mul(self.rel_emb(r1), self.rel_emb(r2))
            rel_embed = rel_embed.view(bs, sl, -1)

        if self.ablation in ('no_qa', 'no_rel', 'no_2hop_qa'):
            concat = rel_embed if self.ablation in ('no_qa', 'no_2hop_qa') else qa_emb
        else:
            concat = torch.cat((qa_emb, rel_embed), -1)

        if self.ablation in ('encode_qas',):
            sent_vecs_expanded = sent_vecs.unsqueeze(1).expand(bs, sl, -1)
            concat = torch.cat((concat, sent_vecs_expanded), -1)

        qars_vecs = self.mlp(concat)
        qars_vecs = self.activation(qars_vecs)

        if self.ablation in ('multihead_pool', 'att_pool'):
            pooled_vecs, att_scores = self.attention(sent_vecs, qars_vecs, mask)
        else:
            qars_vecs = qars_vecs.masked_fill(mask.unsqueeze(2).expand_as(qars_vecs), 0)
            pooled_vecs = qars_vecs.sum(1) / (~mask).float().sum(1).unsqueeze(1).float().to(qars_vecs.device)
            att_scores = None

        if self.ablation == 'no_kg':
            pooled_vecs[:] = 0

        logits = self.hid2out(self.dropout_m(torch.cat((path_embedding, pooled_vecs, sent_vecs), 1)))
        return logits, att_scores

class PromptKGEncoder(nn.Module):

    def __init__(self, concept_num, concept_dim, relation_num, relation_dim, sent_dim, concept_in_dim,
                 hidden_size, num_hidden_layers, kg_enc_out_size, fc_size, num_fc_layers, dropout,
                 pretrained_concept_emb=None, pretrained_relation_emb=None, freeze_ent_emb=True,
                 init_range=0, ablation=None, use_contextualized=False, emb_scale=1.0, path_embedding_dim=768):

        super().__init__()
        self.init_range = init_range
        self.relation_num = relation_num
        self.ablation = ablation

        self.rel_emb = nn.Embedding(relation_num, relation_dim)
        self.concept_emb = CustomizedEmbedding(concept_num=concept_num, concept_out_dim=concept_dim,
                                               use_contextualized=use_contextualized, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                               scale=emb_scale)

        encoder_dim = concept_dim * 2 + relation_dim

        self.mlp = MLP(encoder_dim, hidden_size * 2, hidden_size, num_hidden_layers, dropout,
                       batch_norm=False, layer_norm=True)

        self.dropout_m = nn.Dropout(dropout)
        if self.ablation == "no_dynamic_kg":
            print("Not use dynamic kg generated by gpt2")
            self.hid2out = MLP(hidden_size, fc_size, kg_enc_out_size, num_fc_layers, dropout,
                               batch_norm=False, layer_norm=True)
            # self.mlp = MLP(encoder_dim, kg_enc_out_size*2, kg_enc_out_size, num_hidden_layers, dropout,
            #                    batch_norm=False, layer_norm=True)
        else:
            self.hid2out = MLP(path_embedding_dim + hidden_size, fc_size, kg_enc_out_size, num_fc_layers, dropout,
                               batch_norm=False, layer_norm=True)

        self.activation = GELU()

        if self.init_range > 0:
            self.apply(self._init_weights)

        if pretrained_relation_emb is not None:
            self.rel_emb.weight.data.copy_(pretrained_relation_emb)


        if pretrained_concept_emb is not None:
            self.concept_emb.emb.weight.data.copy_(pretrained_concept_emb)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, path_embedding, sent_vecs, qa_ids, rel_ids, num_tuples, emb_data=None):
        """
        sent_vecs: tensor of shape (batch_size, d_sent)
        qa_ids: tensor of shape (batch_size, max_tuple_num, 2)
        rel_ids: tensor of shape (batch_size, max_tuple_num)
        num_tuples: tensor of shape (batch_size,)
        (emb_data: tensor of shape (batch_size, max_cpt_num, emb_dim))
        """

        bs, sl, _ = qa_ids.size()
        mask = torch.arange(sl, device=qa_ids.device) >= num_tuples.unsqueeze(1) # (bs, sl)
        # TODO: ??
        mask[mask.all(1), 0] = 0  # a temporary solution for instances that have no qar-pairs

        qa_emb = self.concept_emb(qa_ids.view(bs, -1), emb_data).view(bs, sl, -1)
        rel_embed = self.rel_emb(rel_ids)

        # TODO: ??
        if self.ablation not in ('no_factor_mul',):
            n_1hop_rel = int(np.sqrt(self.relation_num))
            assert n_1hop_rel * (n_1hop_rel + 1) == self.relation_num
            rel_ids = rel_ids.view(bs * sl)
            twohop_mask = rel_ids >= n_1hop_rel
            twohop_rel = rel_ids[twohop_mask] - n_1hop_rel
            r1, r2 = twohop_rel // n_1hop_rel, twohop_rel % n_1hop_rel
            assert (r1 >= 0).all() and (r2 >= 0).all() and (r1 < n_1hop_rel).all() and (r2 < n_1hop_rel).all()
            rel_embed = rel_embed.view(bs * sl, -1)
            rel_embed[twohop_mask] = torch.mul(self.rel_emb(r1), self.rel_emb(r2))
            rel_embed = rel_embed.view(bs, sl, -1)

        concat = torch.cat((qa_emb, rel_embed), -1) # (bs, sl, hid)

        qars_vecs = self.mlp(concat)
        # qars_vecs = self.activation(qars_vecs)

        qars_vecs = qars_vecs.masked_fill(mask.unsqueeze(2).expand_as(qars_vecs), 0)
        pooled_vecs = qars_vecs.sum(1) / (~mask).float().sum(1).unsqueeze(1).float().to(qars_vecs.device) # (bs, hid)
        att_scores = None

        if self.ablation == "no_dynamic_kg":
            logits = self.hid2out(self.dropout_m(pooled_vecs)) # (bs, hid)
        else:
            logits = self.hid2out(self.dropout_m(torch.cat((path_embedding, pooled_vecs), 1))) # (bs, hid)
        return logits

class SoftPromptEncoder(nn.Module):

    def __init__(self, args, init_range, embed_size):

        super().__init__()
        self.args = args
        self.init_range = init_range
        self.hidden_size = embed_size
        self.prompt_token_num = self.args.prompt_token_num

        self.prompt_embeddings = torch.nn.Embedding(self.prompt_token_num, embed_size)
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=int(self.hidden_size/2),
                                       num_layers=2,
                                       dropout=0.1,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))

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

    def forward(self, device):
        replace_embeds = self.prompt_embeddings(
            torch.LongTensor(list(range(self.prompt_token_num))).to(device)) # (num_promt_token, embed_size)

        if self.args.using_lstm_mlp:
            replace_embeds = replace_embeds.unsqueeze(0) # (1, num_promt_token, embed_size)

            if self.args.lstm_split:
                ### not split ###
                if self.args.pattern_type == 0: ### _ _ _ q c _ _ _ ###
                    sep = int(self.prompt_token_num / 2)
                    replace_embeds_1 = self.lstm_head(replace_embeds[:, 0:sep, :])[0]
                    replace_embeds_2 = self.lstm_head(replace_embeds[:, sep:,:])[0]
                    replace_embeds = torch.cat((replace_embeds_1, replace_embeds_2), dim=1)
                elif self.args.pattern_type == 1: ### _ _ _ q _ _ _ c _ _ _###
                    sep = int(self.prompt_token_num/3)
                    replace_embeds_1 = self.lstm_head(replace_embeds[:, 0:sep, :])[0]
                    replace_embeds_2 = self.lstm_head(replace_embeds[:, sep:int(2*sep),:])[0]
                    replace_embeds_3 = self.lstm_head(replace_embeds[:, int(2*sep):, :])[0]
                    replace_embeds = torch.cat((replace_embeds_1, replace_embeds_2, replace_embeds_3), dim=1)
                elif self.args.pattern_type == 5: ### _ _ _ q c _ ###
                    sep = self.prompt_token_num-1
                    replace_embeds_1 = self.lstm_head(replace_embeds[:, 0:sep, :])[0]
                    replace_embeds_2 = self.lstm_head(replace_embeds[:, sep:, :])[0]
                    replace_embeds = torch.cat((replace_embeds_1, replace_embeds_2), dim=1)

            else                                                                                                                                                            :
                replace_embeds = self.lstm_head(replace_embeds)[0]  # [1, num_promt_token, 2 * hidden_dim]

            if self.prompt_token_num == 1:
                replace_embeds = self.mlp_head(replace_embeds) # [1, 2, hid_dim]
            else:
                replace_embeds = self.mlp_head(replace_embeds).squeeze() # [num_promt_token, hid_dim]

        return replace_embeds
