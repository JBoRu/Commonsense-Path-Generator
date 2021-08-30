import random

from transformers import *

from modeling.kg_encoder import *
from modeling.models import LMRelationNet, PromptLMRelationNet, PromptWithClassifyLMRelationNet
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

def evaluate_accuracy_prompt_with_classify_head(eval_set, model, type=None):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for sample_ids, qids, labels, *input_data,  prompt_data in eval_set:
            # (bs*5, 2)
            logits, _ = model(*input_data, prompt_data=prompt_data, sample_ids=sample_ids, type=type)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
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

    # find relations between question entities and answer entities
    find_relational_paths(args.cpnet_vocab_path, args.cpnet_graph_path, args.train_concepts, args.train_rel_paths, args.nprocs, args.use_cache)
    find_relational_paths(args.cpnet_vocab_path, args.cpnet_graph_path, args.dev_concepts, args.dev_rel_paths, args.nprocs, args.use_cache)
    if args.test_statements is not None:
        find_relational_paths(args.cpnet_vocab_path, args.cpnet_graph_path, args.test_concepts, args.test_rel_paths, args.nprocs, args.use_cache)

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
        if args.input_format in ['soft-prompt','p-tuning-GPT']:
            freeze_net(model.decoder)
        elif args.input_format in ['p-tuning-GPT-classify']:
            freeze_net(model.decoder)
            freeze_net(model.classify_head)
        else:
            freeze_net(model.decoder.rel_emb)
            freeze_net(model.decoder.concept_emb)
    else:
        if args.input_format in ['soft-prompt','p-tuning-GPT']:
            unfreeze_net(model.decoder)
        elif args.input_format in ['p-tuning-GPT-classify']:
            unfreeze_net(model.decoder)
            unfreeze_net(model.classify_head)
        else:
            unfreeze_net(model.decoder.rel_emb)
            unfreeze_net(model.decoder.concept_emb)

def train(args):
    # print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

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
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1))
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

    # load concrpt relation embeddings
    rel_emb = np.load(args.rel_emb_path)
    rel_emb = np.concatenate((rel_emb, -rel_emb), 0)
    rel_emb = cal_2hop_rel_emb(rel_emb)
    rel_emb = torch.tensor(rel_emb)
    relation_num, relation_dim = rel_emb.size(0), rel_emb.size(1)
    print('| num_concepts: {} | num_relations: {} |'.format(concept_num, relation_num))

    device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() else 'cpu')

    # load path embeddings generated by gpt
    path_embedding_path = os.path.join('./path_embeddings/', args.dataset, 'path_embedding.pickle')

    # initialize dataloader
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
                                      test_node_features_path=args.test_node_features, node_feature_type=args.node_feature_type)

    ###################################################################################################
    #   Build model                                                                                   #
    ###################################################################################################

    lstm_config = get_lstm_config_from_args(args)
    if args.input_format in ['p-tuning', 'hard-prompt', 'soft-prompt', 'p-tuning-GPT']:
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
    elif args.input_format in ['p-tuning-GPT-classify']:
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
    elif args.input_format == 'path-generate':
        model = LMRelationNet(model_name=args.encoder, from_checkpoint=args.from_checkpoint, concept_num=concept_num, concept_dim=relation_dim,
                          relation_num=relation_num, relation_dim=relation_dim,
                          concept_in_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim),
                          hidden_size=args.mlp_dim, num_hidden_layers=args.mlp_layer_num, num_attention_heads=args.att_head_num,
                          fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
                          pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb, freeze_ent_emb=args.freeze_ent_emb,
                          init_range=args.init_range, ablation=args.ablation, use_contextualized=use_contextualized,
                          emb_scale=args.emb_scale, encoder_config=lstm_config)

    try:
        model.to(device)
    except RuntimeError as e:
        print(e)
        print('best dev acc: 0.0 (at epoch 0)')
        print('final test acc: 0.0')
        print()
        return

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.input_format in ['p-tuning-GPT-classify']:
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
    elif 'warmup' in args.lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    print('parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
        else:
            print('\t{:45}\tfixed\t{}'.format(name, param.size()))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print('-' * 71)
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    # freeze_net(model.encoder)
    # try:
    rel_grad = []
    linear_grad = []
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
                else:
                    logits, mlm_labels = model(*[x[a:b] for x in input_data], prompt_data=[x[a:b] for x in prompt_data], sample_ids=sample_ids[a:b], type='train')

                if args.loss == 'margin_rank':
                    num_choice = logits.size(1)
                    flat_logits = logits.view(-1)
                    correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
                    correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
                    wrong_logits = flat_logits[correct_mask == 0]  # of length batch_size*(num_choice-1)
                    y = wrong_logits.new_ones((wrong_logits.size(0),))
                    loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
                elif args.loss == 'cross_entropy':
                    if args.input_format in ['p-tuning', 'hard-prompt', 'soft-prompt', 'p-tuning-GPT']:
                        loss = loss_func(logits, mlm_labels) # (bs, 2) (bs)
                    elif args.input_format in ['path-generate', 'p-tuning-GPT-classify']:
                        loss = loss_func(logits, labels[a:b])

                loss = loss * (b - a) / bs
                loss.backward()
                total_loss += loss.item()

            if (global_step + 1) % args.grad_step == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} |  kg_enc_lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_last_lr()[2], total_loss, ms_per_batch))
                # print('| rel_grad: {:1.2e} | linear_grad: {:1.2e} |'.format(sum(rel_grad) / len(rel_grad), sum(linear_grad) / len(linear_grad)))
                total_loss = 0
                rel_grad = []
                linear_grad = []
                start_time = time.time()
            global_step += 1

        model.eval()
        if args.input_format in ['p-tuning', 'hard-prompt', 'soft-prompt', 'p-tuning-GPT']:
            dev_acc = evaluate_accuracy_prompt(dataset.dev(), model, type='dev')
            test_acc = evaluate_accuracy_prompt(dataset.test(), model, type='test') if dataset.test_size() > 0 else 0.0
        elif args.input_format in ['p-tuning-GPT-classify']:
            dev_acc = evaluate_accuracy_prompt_with_classify_head(dataset.dev(), model, type='dev')
            test_acc = evaluate_accuracy_prompt_with_classify_head(dataset.test(), model, type='test') if dataset.test_size() > 0 else 0.0
        elif args.input_format == 'path-generate':
            dev_acc = evaluate_accuracy(dataset.dev(), model)
            test_acc = evaluate_accuracy(dataset.test(), model) if dataset.test_size() > 0 else 0.0

        print('-' * 71)
        print('| epoch {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, dev_acc, test_acc))
        print('-' * 71)
        with open(log_path, 'a') as fout:
            fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            final_test_acc = test_acc
            best_dev_epoch = epoch_id
            if args.save_model == 1:
                torch.save([model, args], model_path)
                print(f'model saved to {model_path}')

        model.train()
        start_time = time.time()
        if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            break

    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev acc: {:.4f} (at epoch {})'.format(best_dev_acc, best_dev_epoch))
    print('final test acc: {:.4f}'.format(final_test_acc))
    print()


def eval(args):
    raise NotImplementedError()


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

if __name__ == '__main__':
    main()
