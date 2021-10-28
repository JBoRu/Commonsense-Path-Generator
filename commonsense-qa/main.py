import random

import numpy as np
import torch
import copy
from transformers import *

from modeling.kg_encoder import *
from modeling.models import *
# LMRelationNet, PromptLMRelationNet, PromptWithClassifyLMRelationNet, PromptWithGenerateLMRelationNet, PromptWithKCRClassify, PromptWithKCRGenerate
from modeling.data_loader import *
from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *
from utils.relpath_utils import find_relational_paths

def get_node_feature_encoder(encoder_name):
    return encoder_name.replace('-cased', '-uncased')


def cal_2hop_rel_emb(rel_emb):
    n_rel = rel_emb.shape[0]
    u, v = np.meshgrid(np.arange(n_rel), np.arange(n_rel))
    expanded = rel_emb[v.reshape(-1)] + rel_emb[u.reshape(-1)]
    return np.concatenate([rel_emb, expanded], 0)


def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data, prompt_data in eval_set:
            logits, _ = model(*input_data)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples

def evaluate_accuracy_prompt(eval_set, model, type=None):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for sample_ids, qids, labels, *input_data,  prompt_data in eval_set:
            # (bs*5, 2)
            logits, _ = model(*input_data, prompt_data=prompt_data, sample_ids=sample_ids, type=type)
            logits = logits[:, 1:2] # (bs*5) get the logit of YES
            logits = logits.view(-1, 5) # (bs, 5)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples

def evaluate_accuracy_prompt_for_concat_choices(eval_set, model, type=None):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for sample_ids, qids, labels, *input_data,  prompt_data in eval_set:
            # (bs*1, 5)
            logits, _ = model(*input_data, prompt_data=prompt_data, sample_ids=sample_ids, type=type)
            logits = logits.view(-1, 5) # (bs, 5)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples

def evaluate_accuracy_prompt_with_classify_head(eval_set, model, type=None, num_choice=5):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for sample_ids, qids, labels, *input_data,  prompt_data in eval_set:
            # (bs*5, 2)
            logits, _ = model(*input_data, prompt_data=prompt_data, sample_ids=sample_ids, type=type)
            # logits = logits.view(-1,5)
            logits = logits.view(-1, num_choice)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples

def evaluate_accuracy_prompt_with_generate(eval_set, model, type=None):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for sample_ids, qids, labels, *input_data,  prompt_data in eval_set:
            # (bs*5, 2)
            predict_logits, target_ids, label_mask, eval_logits = model(*input_data, prompt_data=prompt_data, sample_ids=sample_ids, type=type)
            logits = eval_logits
            logits = logits[:, 1:2]  # (bs*5) get the logit of YES
            logits = logits.view(-1, 5)  # (bs, 5)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples

def evaluate_accuracy_prompt_with_classify_head_for_cat_two_cho(eval_set, model, type=None):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for sample_ids, qids, labels, *input_data,  prompt_data in eval_set:
            # (bs*10, 2)
            logits, _ = model(*input_data, prompt_data=prompt_data, sample_ids=sample_ids, type=type)
            logits = logits.view(-1,10,2) # (bs, 10, 2)
            logits_abs = torch.abs(logits[:, :, 0] - logits[:, :, 1])
            logits = logits.argmax(dim=-1) # (bs, 10)
            num_sap = logits.shape[0]
            pred = []
            labels = labels.cpu().numpy().tolist()
            for sid in range(num_sap):
                logit = logits[sid]
                logit_abs = logits_abs[sid]
                logit_abs = logit_abs.cpu().numpy().tolist()
                win_count = [0] * 5  # [0, 0, 0, 0, 0]
                idx = 0
                winer_dict = {}
                diff_dict = {}
                for i in range(5):
                    for j in range(5):
                        if j > i:
                            key = str(i) + "_" + str(j)
                            if logit[idx] == 0:
                                win_count[i] += 1
                                winer_dict[key] = i
                            else:
                                win_count[j] += 1
                                winer_dict[key] = j
                            diff_dict[key] = logit_abs[idx]
                            idx += 1
                win_count = np.array(win_count)
                max = np.max(win_count)
                flag = (win_count == max)
                win = []
                for idx, f in enumerate(flag):
                    if f:
                        win.append(idx)
                if len(win) == 1:
                    pred.append(win[0])
                elif len(win) == 2:
                    key = str(win[0]) + "_" + str(win[1])
                    pred.append(winer_dict[key])
                elif len(win) == 3:
                    key1 = str(win[0]) + "_" + str(win[1])
                    key2 = str(win[1]) + "_" + str(win[2])
                    key3 = str(win[0]) + "_" + str(win[2])
                    s1, w1 = diff_dict[key1], winer_dict[key1]
                    s2, w2 = diff_dict[key2], winer_dict[key2]
                    s3, w3 = diff_dict[key3], winer_dict[key3]
                    score = np.array([s1, s2, s3])
                    max_s = np.max(score)
                    flag_s = (score == max_s)
                    win_s = []
                    for idx, f in enumerate(flag_s):
                        if f:
                            win_s.append(idx)
                    assert len(win_s) == 1, "Vote error!"
                    if win_s[0] == 0:
                        pred.append(w1)
                    elif win_s[0] == 1:
                        pred.append(w2)
                    else:
                        pred.append(w3)
            for p, l in zip(pred, labels):
                if p == l:
                    n_correct += 1
            n_samples += len(labels)
    return n_correct / n_samples


def pred_to_file(eval_set, model, output_path):
    model.eval()
    fw = open(output_path, 'w')
    with torch.no_grad():
        for qids, labels, *input_data in eval_set:
            logits, _ = model(*input_data)
            for qid, pred_label in zip(qids, logits.argmax(1)):
                fw.write('{},{}\n'.format(qid, chr(ord('A') + pred_label.item())))
    fw.close()

def main():
    parser = get_parser()
    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)

    # # find relations between question entities and answer entities
    # find_relational_paths(args.cpnet_vocab_path, args.cpnet_graph_path, args.train_concepts, args.train_rel_paths, args.nprocs, args.use_cache)
    # find_relational_paths(args.cpnet_vocab_path, args.cpnet_graph_path, args.dev_concepts, args.dev_rel_paths, args.nprocs, args.use_cache)
    # if args.test_statements is not None:
    #     find_relational_paths(args.cpnet_vocab_path, args.cpnet_graph_path, args.test_concepts, args.test_rel_paths, args.nprocs, args.use_cache)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)
    elif args.mode == 'pred':
        pred(args)
    else:
        raise ValueError('Invalid mode')

def freeze_and_unfreeze_net(model, args):
    if args.freeze_enc:
        freeze_net(model.encoder)
    else:
        unfreeze_net(model.encoder)

    if args.freeze_dec: # refer to decoder embeddings
        if args.input_format in ['soft_prompt_p_tuning','GPT_kg_generator_as_prompt', 'p-tuning-GPT-generate']:
            freeze_net(model.decoder)
        elif args.input_format in ['soft_prompt_p_tuning_classify']:
            freeze_net(model.decoder)
            freeze_net(model.classify_head)
        else:
            freeze_net(model.decoder.rel_emb)
            freeze_net(model.decoder.concept_emb)
    else:
        if args.input_format in ['soft_prompt_p_tuning', 'GPT_kg_generator_as_prompt', 'p-tuning-GPT-generate']:
            unfreeze_net(model.decoder)
        elif args.input_format in ['soft_prompt_p_tuning_classify']:
            unfreeze_net(model.decoder)
            unfreeze_net(model.classify_head)
        else:
            unfreeze_net(model.decoder.rel_emb)
            unfreeze_net(model.decoder.concept_emb)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args):
    # print(args)

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,train_acc,dev_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################

    if 'lm' in args.ent_emb:
        print('Using contextualized embeddings for concepts')
        use_contextualized, cp_emb = True, None
    else:
        use_contextualized = False

    # load concept entity embeddings
    # cp_emb = [np.load(path) for path in args.ent_emb_paths]
    # cp_emb = torch.tensor(np.concatenate(cp_emb, 1))
    # concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

    # load concrpt relation embeddings
    # rel_emb = np.load(args.rel_emb_path)
    # rel_emb = np.concatenate((rel_emb, -rel_emb), 0)
    # rel_emb = cal_2hop_rel_emb(rel_emb)
    # rel_emb = torch.tensor(rel_emb)
    # relation_num, relation_dim = rel_emb.size(0), rel_emb.size(1)
    # print('| num_concepts: {} | num_relations: {} |'.format(concept_num, relation_num))

    device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() else 'cpu')

    # load path embeddings generated by gpt
    path_embedding_path = os.path.join('./path_embeddings/', args.dataset, 'path_embedding.pickle')

    # initialize dataloader
    if args.experiment_base == "p-tuning":
        dataset = LMRelationNetDataLoader(args, args.train_statements, args.dev_statements, args.test_statements,
                                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                device=device, model_name=args.encoder, max_seq_length=args.max_seq_len,
                                is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids)
    elif args.experiment_base in ["kcr","continue_train_with_kcr"]:
        dataset = KCRDataLoader(args, args.train_statements, args.dev_statements, args.test_statements,
                                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                device=device, model_name=args.encoder, max_seq_length=args.max_seq_len,
                                is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids)
    elif args.experiment_base == "inter_pretrain_nli":
        dataset = NLIDataLoader(args, args.train_statements, args.dev_statements, args.test_statements,
                                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                device=device, model_name=args.encoder, max_seq_length=args.max_seq_len, num_label=3)
    elif args.experiment_base == "mixed_nli_csqa_kcr":
        dataset = MixedDataLoader(args, args.train_statements, args.dev_statements, args.test_statements,
                                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                device=device, model_name=args.encoder, max_seq_length=args.max_seq_len, num_label=5,
                                is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids)

    ###################################################################################################
    #   Build model                                                                                   #
    ###################################################################################################

    lstm_config = get_lstm_config_from_args(args)
    if args.experiment_base == "p-tuning":
        if args.input_format in ['pg_kg_enc_as_prompt', 'manual_hard_prompt', 'soft_prompt_p_tuning', 'GPT_kg_generator_as_prompt']:
            model = PromptLMRelationNet(args=args, model_name=args.encoder, label_list_len=2, from_checkpoint=args.from_checkpoint,
                              concept_num=concept_num, concept_dim=relation_dim,
                              relation_num=relation_num, relation_dim=relation_dim,
                              concept_in_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim),
                              hidden_size=args.mlp_dim, num_hidden_layers=args.mlp_layer_num, prompt_token_num=args.prompt_token_num,
                              fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
                              pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb, freeze_ent_emb=args.freeze_ent_emb,
                              init_range=args.init_range, ablation=args.ablation, use_contextualized=use_contextualized,
                              emb_scale=args.emb_scale)
            freeze_and_unfreeze_net(model, args)
        elif args.input_format in ['soft_prompt_p_tuning_classify']:
            model = PromptWithClassifyLMRelationNet(args=args, model_name=args.encoder, label_list_len=2,
                                        from_checkpoint=args.from_checkpoint,
                                        concept_num=concept_num, concept_dim=relation_dim,
                                        relation_num=relation_num, relation_dim=relation_dim,
                                        concept_in_dim=(
                                            dataset.get_node_feature_dim() if use_contextualized else concept_dim),
                                        hidden_size=args.mlp_dim, num_hidden_layers=args.mlp_layer_num,
                                        prompt_token_num=args.prompt_token_num,
                                        fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
                                        pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb,
                                        freeze_ent_emb=args.freeze_ent_emb,
                                        init_range=args.init_range, ablation=args.ablation,
                                        use_contextualized=use_contextualized,
                                        emb_scale=args.emb_scale)
            freeze_and_unfreeze_net(model, args)
        elif args.input_format in ['p-tuning-GPT-generate']:
            model = PromptWithGenerateLMRelationNet(args=args, model_name=args.encoder, label_list_len=2,
                                        from_checkpoint=args.from_checkpoint,
                                        concept_num=concept_num, concept_dim=relation_dim,
                                        relation_num=relation_num, relation_dim=relation_dim,
                                        concept_in_dim=(
                                            dataset.get_node_feature_dim() if use_contextualized else concept_dim),
                                        hidden_size=args.mlp_dim, num_hidden_layers=args.mlp_layer_num,
                                        prompt_token_num=args.prompt_token_num,
                                        fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
                                        pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb,
                                        freeze_ent_emb=args.freeze_ent_emb,
                                        init_range=args.init_range, ablation=args.ablation,
                                        use_contextualized=use_contextualized,
                                        emb_scale=args.emb_scale)
            freeze_and_unfreeze_net(model, args)
        elif args.input_format == 'path-generate':
            model = LMRelationNet(model_name=args.encoder, from_checkpoint=args.from_checkpoint, concept_num=concept_num, concept_dim=relation_dim,
                              relation_num=relation_num, relation_dim=relation_dim,
                              concept_in_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim),
                              hidden_size=args.mlp_dim, num_hidden_layers=args.mlp_layer_num, num_attention_heads=args.att_head_num,
                              fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
                              pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb, freeze_ent_emb=args.freeze_ent_emb,
                              init_range=args.init_range, ablation=args.ablation, use_contextualized=use_contextualized,
                              emb_scale=args.emb_scale, encoder_config=lstm_config)
    elif args.experiment_base == "kcr":
        if args.input_format in ['soft_prompt_p_tuning_classify']:
            if args.concat_choices:
                model = PromptWithKCRWithConcatChoicesClassify(args=args, model_name=args.encoder, label_list_len=2,
                                              from_checkpoint=args.from_checkpoint,
                                              prompt_token_num=args.prompt_token_num,
                                              init_range=args.init_range)
            elif args.concat_two_choices:
                model = PromptWithKCRWithConcatTwoChoicesClassify(args=args, model_name=args.encoder, label_list_len=2,
                                                               from_checkpoint=args.from_checkpoint,
                                                               prompt_token_num=args.prompt_token_num,
                                                               init_range=args.init_range)
            else:
                model = PromptWithKCRClassify(args=args, model_name=args.encoder, label_list=[0, 1, 2, 3, 4],
                                                from_checkpoint=args.from_checkpoint,
                                                prompt_token_num=args.prompt_token_num,
                                                init_range=args.init_range)
            freeze_and_unfreeze_net(model, args)
        elif args.input_format in ['soft_prompt_p_tuning']:
            model = PromptWithKCRGenerate(args=args, model_name=args.encoder, label_list_len=2,
                                          from_checkpoint=args.from_checkpoint,
                                          prompt_token_num=args.prompt_token_num,
                                          init_range=args.init_range)
            freeze_and_unfreeze_net(model, args)
    elif args.experiment_base == "inter_pretrain_nli":
        if args.input_format in ['soft_prompt_p_tuning_classify']:
            # model = PromptWithNLIClassify(args=args, model_name=args.encoder, label_list_len=2,
            #                                 from_checkpoint=args.from_checkpoint,
            #                                 prompt_token_num=args.prompt_token_num,
            #                                 init_range=args.init_range)
            model = PromptWithKCRClassify(args=args, model_name=args.encoder, label_list=[0,1],
                                          from_checkpoint=args.from_checkpoint,
                                          prompt_token_num=args.prompt_token_num,
                                          init_range=args.init_range)
            freeze_and_unfreeze_net(model, args)
    elif args.experiment_base == "continue_train_with_kcr":
        model = PromptWithKCRClassify(args=args, model_name=args.encoder, label_list=[0, 1, 2, 3, 4],
                                      from_checkpoint=args.from_checkpoint,
                                      prompt_token_num=args.prompt_token_num,
                                      init_range=args.init_range)
        checkpoint_path = args.checkpoint_path
        check_model, check_args = torch.load(checkpoint_path, map_location='cpu')
        new_dict = {}
        for k, v in check_model.state_dict().items():
            if k not in model.state_dict():
                print(k)
                continue
            new_dict[k] = v
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print("Load pretrained parameters from ", checkpoint_path)
    elif args.experiment_base == "mixed_nli_csqa_kcr":
        if args.input_format in ['soft_prompt_p_tuning_classify']:
            model = PromptWithKCRClassify(args=args, model_name=args.encoder, label_list=[0,1,2,3,4],
                                          from_checkpoint=args.from_checkpoint,
                                          prompt_token_num=args.prompt_token_num,
                                          init_range=args.init_range)
            freeze_and_unfreeze_net(model, args)

    try:
        model.to(device)
    except RuntimeError as e:
        print(e)
        print('best dev acc: 0.0 (at epoch 0)')
        print('final test acc: 0.0')
        print()
        return

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.input_format in ['soft_prompt_p_tuning_classify']:
        grouped_parameters = [
            {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
            {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.encoder_lr},
            {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
            {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.decoder_lr},
            {'params': [p for n, p in model.classify_head.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
            {'params': [p for n, p in model.classify_head.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.decoder_lr}
        ]
    else:
        grouped_parameters = [
            {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
            {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.encoder_lr},
            {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
            {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.decoder_lr},
        ]

    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)
    if args.lr_schedule == 'fixed':
        scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        print("Using warmup linear with warmup steps: {} of max steps: {}".format(args.warmup_steps, max_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)
    elif args.lr_schedule == 'warmup_cosine_hard_restart':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        warmup_proportion = args.warmup_proportion
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,)
    elif args.lr_schedule == 'warmup_cosine':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        warmup_proportion = args.warmup_proportion
        warmup_steps = int(max_steps*warmup_proportion)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    print('parameters:')
    for name, param in model.named_parameters():
        if ("encoder.layer." not in name) or "encoder.layer.0" in name:
            if param.requires_grad:
                print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
            else:
                print('\t{:45}\tfixed\t{}'.format(name, param.size()))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=args.margin, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    elif args.loss == 'binary_cross_entropy':
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print('-' * 71)
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    rel_grad = []
    linear_grad = []
    # model.eval()
    # if args.input_format in ['soft_prompt_p_tuning_classify']:
    #     if args.concat_two_choices:
    #         dev_acc = evaluate_accuracy_prompt_with_classify_head_for_cat_two_cho(dataset.dev(), model, type='dev')
    #         test_acc = 0.0
    #     else:
    #         dev_acc = evaluate_accuracy_prompt_with_classify_head(dataset.dev(), model, type='dev')
    #         test_acc = evaluate_accuracy_prompt_with_classify_head(dataset.test(), model, type='test') \
    #             if dataset.test_size() > 0 else 0.0
    #
    #     print('| Before training | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(dev_acc, test_acc))

    model.train()
    for epoch_id in range(args.n_epochs):
        # if epoch_id == args.unfreeze_epoch:
        #     print('encoder unfreezed')
        #     unfreeze_net(model.encoder)
        # if epoch_id == args.refreeze_epoch:
        #     print('encoder refreezed')
        #     freeze_net(model.encoder)
        model.train()
        for sample_ids, qids, labels, *input_data, prompt_data in dataset.train():
            optimizer.zero_grad()
            bs = labels.size(0)
            for a in range(0, bs, args.mini_batch_size):
                b = min(a + args.mini_batch_size, bs)
                if args.input_format == 'path-generate':
                    logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                elif args.input_format in ['p-tuning-GPT-generate']:
                    pred_logits, target_ids, label_mask, eval_logits = model(*[x[a:b] for x in input_data], prompt_data=[x[a:b] for x in prompt_data],
                                               sample_ids=sample_ids[a:b], type='train')
                else:
                    logits, mlm_labels = model(*[x[a:b] for x in input_data], prompt_data=[x[a:b] for x in prompt_data],
                                               sample_ids=sample_ids[a:b], type='train')
                    # logits = logits.view(-1, self.label_list_len)

                if args.loss == 'margin_rank':
                    logits = logits.view(-1, args.num_choice)
                    num_choice = logits.size(1)
                    flat_logits = logits.view(-1) # (bs*5,)
                    correct_mask = F.one_hot(labels[a:b], num_classes=num_choice).view(-1)  # (bs*5,)
                    # print("Flat logits: ", flat_logits.shape, "Labels: ", labels[a:b].shape, "One hot mask: ", correct_mask.shape)
                    correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # (batch_size*(num_choice-1),)
                    # print("Correct logits: ", correct_logits.shape)
                    wrong_logits = flat_logits[correct_mask == 0]  # (bs*num_choice-1,)
                    # print("Wrong logits: ", wrong_logits.shape)
                    y = wrong_logits.new_ones((wrong_logits.size(0),)) # (bs*num_choice-1,)
                    # print("Label logits: ", y.shape)
                    loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
                elif args.loss == 'cross_entropy':
                    if args.input_format in ['pg_kg_enc_as_prompt', 'manual_hard_prompt', 'soft_prompt_p_tuning', 'GPT_kg_generator_as_prompt']:
                        loss = loss_func(logits, mlm_labels) # (bs, 2) (bs)
                    elif args.input_format in ['p-tuning-GPT-generate']:
                        loss_func = nn.CrossEntropyLoss(reduction='none')
                        loss = loss_func(pred_logits, target_ids)
                        seq_len = label_mask.shape[-1]
                        loss = loss.view(-1, 5, seq_len)
                        loss = torch.mul(loss, label_mask)
                        loss = torch.sum(loss) / torch.sum(label_mask)
                    elif args.input_format in ['path-generate', 'soft_prompt_p_tuning_classify']:
                        if args.concat_two_choices:
                            loss = loss_func(logits, mlm_labels)
                        else:
                            logits = logits.view(-1, args.num_choice)
                            loss = loss_func(logits, labels[a:b])
                elif args.loss == "binary_cross_entropy":
                    logits = logits.view(-1) #(bs*5)
                    mlm_labels = mlm_labels.float()
                    loss = loss_func(logits, mlm_labels)

                loss = loss * (b - a) / bs
                loss.backward()
                total_loss += loss.item()

            # before = copy.deepcopy(model.encoder.embeddings.weight.data)

            if (global_step + 1) % args.grad_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            # after = copy.deepcopy(model.encoder.embeddings.weight.data)
            #
            # if (after != before).sum():
            #     print("Update!")

            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} |  kg_enc_lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_last_lr()[2], total_loss, ms_per_batch))
                # print('| rel_grad: {:1.2e} | linear_grad: {:1.2e} |'.format(sum(rel_grad) / len(rel_grad), sum(linear_grad) / len(linear_grad)))
                total_loss = 0
                rel_grad = []
                linear_grad = []
                start_time = time.time()
                if args.experiment_base == "inter_pretrain_nli":
                    check_model_path = os.path.join(args.save_dir, 'model_bs_8_global_step_') + str(global_step)
                    torch.save([model, args], check_model_path)
                    print(f'Checkpoint model saved to {check_model_path}')
            global_step += 1

        model.eval()
        if args.input_format in ['pg_kg_enc_as_prompt', 'manual_hard_prompt', 'soft_prompt_p_tuning', 'GPT_kg_generator_as_prompt']:
            if args.concat_choices:
                dev_acc = evaluate_accuracy_prompt_for_concat_choices(dataset.dev(), model, type='dev')
                test_acc = evaluate_accuracy_prompt_for_concat_choices(dataset.test(), model,
                                                    type='test') if dataset.test_size() > 0 else 0.0
            else:
                dev_acc = evaluate_accuracy_prompt(dataset.dev(), model, type='dev')
                test_acc = evaluate_accuracy_prompt(dataset.test(), model, type='test') if dataset.test_size() > 0 else 0.0
        elif args.input_format in ['soft_prompt_p_tuning_classify']:
            if args.concat_two_choices:
                dev_acc = evaluate_accuracy_prompt_with_classify_head_for_cat_two_cho(dataset.dev(), model, type='dev')
                test_acc = 0.0
            else:
                dev_acc = evaluate_accuracy_prompt_with_classify_head(dataset.dev(), model, type='dev',
                                                                      num_choice=args.num_choice)
                test_acc = evaluate_accuracy_prompt_with_classify_head(dataset.test(), model, type='test',
                                                                       num_choice=args.num_choice) \
                                                                                    if dataset.test_size() > 0 else 0.0
        elif args.input_format in ['p-tuning-GPT-generate']:
            dev_acc = evaluate_accuracy_prompt_with_generate(dataset.dev(), model, type='dev')
            test_acc = evaluate_accuracy_prompt_with_generate(dataset.test(), model,
                                                                   type='test') if dataset.test_size() > 0 else 0.0
        elif args.input_format == 'path-generate':
            dev_acc = evaluate_accuracy(dataset.dev(), model)
            test_acc = evaluate_accuracy(dataset.test(), model) if dataset.test_size() > 0 else 0.0

        print('-' * 71)
        print('| epoch {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, dev_acc, test_acc))
        print('-' * 71)
        with open(log_path, 'a') as fout:
            fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))

        if args.save_checkpoint > 0 and (epoch_id+1) % args.save_checkpoint == 0:
            check_model_path = os.path.join(args.save_dir, 'model_bs_8_epoch_') + str(epoch_id)
            torch.save([model, args], check_model_path)
            print(f'Checkpoint model saved to {check_model_path}')

        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            final_test_acc = test_acc
            best_dev_epoch = epoch_id
            if args.save_model == 1:
                torch.save([model, args], model_path)
                print(f'model saved to {model_path}')

        model.train()
        start_time = time.time()
        if args.lr_schedule == 'fixed':
            if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break

    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev acc: {:.4f} (at epoch {})'.format(best_dev_acc, best_dev_epoch))
    print('final test acc: {:.4f}'.format(final_test_acc))
    print()


def eval(args):
    train_pred_path = os.path.join(args.save_dir, 'predictions_train')
    dev_pred_path = os.path.join(args.save_dir, 'predictions_dev')
    test_pred_path = os.path.join(args.save_dir, 'predictions_test')
    model_path = os.path.join(args.save_dir, 'model.pt')
    # model_path = os.path.join(args.save_dir, 'model_bs_8_epoch_2')
    device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() and args.cuda else "cpu")
    print("Load trained model form %s"%(model_path))
    model, old_args = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    path_embedding_path = None
    use_contextualized = False
    # initialize dataloader
    dataset = None
    if args.experiment_base == "p-tuning":
        dataset = LMRelationNetDataLoader(args, path_embedding_path,
                                          args.train_statements, args.train_rel_paths,
                                          args.dev_statements, args.dev_rel_paths,
                                          args.test_statements, args.test_rel_paths,
                                          batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                          device=device,
                                          model_name=args.encoder,
                                          max_tuple_num=args.max_tuple_num, max_seq_length=args.max_seq_len,
                                          is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                          use_contextualized=use_contextualized,
                                          train_adj_path=args.train_adj, dev_adj_path=args.dev_adj,
                                          test_adj_path=args.test_adj,
                                          train_node_features_path=args.train_node_features,
                                          dev_node_features_path=args.dev_node_features,
                                          test_node_features_path=args.test_node_features,
                                          node_feature_type=args.node_feature_type)
    elif args.experiment_base == "kcr":
        dataset = KCRDataLoader(args, args.train_statements, args.dev_statements, args.test_statements,
                                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                device=device, model_name=args.encoder, max_seq_length=args.max_seq_len,
                                is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids)

    print("***** generating model predictions *****")
    print(f'| dataset: {args.dataset} | save_dir: {args.save_dir} |')
    output_path = dev_pred_path
    data_loader = dataset.dev()
    n_correct = 0.0
    n_samples = 0.0

    with torch.no_grad(), open(output_path, 'w') as fout:
        for id, (sample_ids, qids, labels, *input_data, prompt_data) in enumerate(data_loader):
            # (bs*10, 2)
            logits, mlm_labels = model(*input_data, prompt_data=prompt_data, sample_ids=sample_ids, type='dev')
            if args.concat_two_choices:
                n_correct, n_samples = inference_for_pairwise_ranking(qids, logits, labels, fout, n_correct, n_samples)
            else:
                n_correct, n_samples = inference_for_classify_ranking(qids, logits, labels, fout, n_correct, n_samples, args.num_choice)

        result = n_correct / n_samples
        result = "Accuracy:%.4f\n" % (result)
        fout.write(result)

    print(f'predictions saved to {output_path}')
    print('***** prediction done *****')

def pred(args):

    dev_pred_path = os.path.join(args.save_dir, 'predictions_dev.csv')
    test_pred_path = os.path.join(args.save_dir, 'predictions_test.csv')
    model_path = os.path.join(args.save_dir, 'model.pt')
    device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() and args.cuda else "cpu")
    model, old_args = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    if 'lm' in args.ent_emb:
        print('Using contextualized embeddings for concepts')
        use_contextualized, cp_emb = True, None
    else:
        use_contextualized = False

    path_embedding_path = os.path.join('./path_embeddings/', args.dataset, 'path_embedding.pickle')

    dataset = LMRelationNetDataLoaderForPred(path_embedding_path, old_args.train_statements, old_args.train_rel_paths,
                                      old_args.dev_statements, old_args.dev_rel_paths,
                                      old_args.test_statements, old_args.test_rel_paths,
                                      batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=device,
                                      model_name=old_args.encoder,
                                      max_tuple_num=old_args.max_tuple_num, max_seq_length=old_args.max_seq_len,
                                      is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                      use_contextualized=use_contextualized,
                                      train_adj_path=args.train_adj, dev_adj_path=args.dev_adj, test_adj_path=args.test_adj,
                                      train_node_features_path=args.train_node_features, dev_node_features_path=args.dev_node_features,
                                      test_node_features_path=args.test_node_features, node_feature_type=args.node_feature_type)
    print("***** generating model predictions *****")
    print(f'| dataset: {old_args.dataset} | save_dir: {args.save_dir} |')

    for output_path, data_loader in ([(test_pred_path, dataset.test())] if dataset.test_size() > 0 else []):
        with torch.no_grad(), open(output_path, 'w') as fout:
            for qids, labels, *input_data in tqdm(data_loader):
                logits, _ = model(*input_data)
                for qid, pred_label in zip(qids, logits.argmax(1)):
                    # fout.write('{},{}\n'.format(qid, chr(ord('A') + pred_label.item())))
                    fout.write('{}\n'.format(pred_label))
        print(f'predictions saved to {output_path}')
    print('***** prediction done *****')

def inference_for_pairwise_ranking(qids, logits, labels, fout, n_correct, n_samples):
    logits = logits.view(-1, 10, 2)  # (bs, 10, 2)
    logits_abs = torch.abs(logits[:, :, 0] - logits[:, :, 1])
    logits_abs = logits_abs.cpu().numpy().tolist()
    logits = logits.argmax(dim=-1)  # (bs, 10)
    num_sap = logits.shape[0]
    preds = []
    labels = labels.cpu().numpy().tolist()
    pred_logits = np.zeros((num_sap, 5, 5))
    for sid in range(num_sap):
        win_count = [0] * 5  # [0, 0, 0, 0, 0]
        idx = 0
        winer_dict = {}
        diff_dict = {}
        for i in range(5):
            for j in range(5):
                if j > i:
                    score_diff = logits_abs[sid][idx] if logits[sid][idx] == 0 else (-1.0) * logits_abs[sid][idx]
                    pred_logits[sid][i][j] = score_diff
                    # idx += 1
                    key = str(i) + "_" + str(j)
                    if logits[sid][idx] == 0:
                        win_count[i] += 1
                        winer_dict[key] = i
                    else:
                        win_count[j] += 1
                        winer_dict[key] = j
                    diff_dict[key] = logits_abs[sid][idx]
                    idx += 1
        win_count = np.array(win_count)
        max = np.max(win_count)
        flag = (win_count == max)
        win = []
        for idx, f in enumerate(flag):
            if f:
                win.append(idx)
        if len(win) == 1:
            preds.append(win[0])
        elif len(win) == 2:
            key = str(win[0]) + "_" + str(win[1])
            preds.append(winer_dict[key])
        elif len(win) == 3:
            key1 = str(win[0]) + "_" + str(win[1])
            key2 = str(win[1]) + "_" + str(win[2])
            key3 = str(win[0]) + "_" + str(win[2])
            s1, w1 = diff_dict[key1], winer_dict[key1]
            s2, w2 = diff_dict[key2], winer_dict[key2]
            s3, w3 = diff_dict[key3], winer_dict[key3]
            score = np.array([s1, s2, s3])
            max_s = np.max(score)
            flag_s = (score == max_s)
            win_s = []
            for idx, f in enumerate(flag_s):
                if f:
                    win_s.append(idx)
            assert len(win_s) == 1, "Vote error!"
            if win_s[0] == 0:
                preds.append(w1)
            elif win_s[0] == 1:
                preds.append(w2)
            else:
                preds.append(w3)

    for p, l in zip(preds, labels):
        if p == l:
            n_correct += 1
    n_samples += len(labels)

    for idx, (qid, label, pred) in enumerate(zip(qids, labels, preds)):
        title = "Question: " + str(qid) + " Label: " + str(label) + " Pred: " + str(pred)
        fout.write("{}\n".format(title))
        for i in range(5):
            line_logit = pred_logits[idx][i]
            line_logit = "\t".join(["%.3f" % l for l in line_logit])
            line_logit = "\t" + line_logit + "\n"
            fout.write(line_logit)
        fout.write("\n")

    return n_correct, n_samples

def inference_for_classify_ranking(qids, logits, labels, fout, n_correct, n_samples, num_choice):
    logits = logits.view(-1, num_choice)
    preds = logits.argmax(1)
    preds = preds.cpu().numpy().tolist()
    n_correct += (logits.argmax(1) == labels).sum().item()
    n_samples += labels.size(0)
    labels = labels.cpu().numpy().tolist()

    for idx, (qid, label, pred) in enumerate(zip(qids, labels, preds)):
        title = str(qid) + "\t" + str(label) + "\t" + str(pred)
        fout.write("{}\n".format(title))

    return n_correct, n_samples

if __name__ == '__main__':
    main()
