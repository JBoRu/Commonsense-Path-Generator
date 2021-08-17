from modeling.text_encoder import TextEncoder
from utils.layers import *

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
