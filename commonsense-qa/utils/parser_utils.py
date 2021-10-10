from utils.utils_ import *
from multiprocessing import cpu_count
ENCODER_DEFAULT_LR = {
    'default': 1e-3,
    'csqa': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
        'albert-xxlarge-v2': 1e-5
    },
    'obqa': {
        'lstm': 3e-4,
        'openai-gpt': 3e-5,
        'bert-base-cased': 1e-4,
        'bert-large-cased': 1e-4,
        'roberta-large': 1e-5,
        'albert-xxlarge-v2': 1e-5
    }
}

DATASET_LIST = ['csqa', 'obqa', 'socialiqa', 'phys']

DATASET_SETTING = {
    'csqa': 'inhouse',
    'obqa': 'official',
    'socialiqa': 'official',
    'phys': 'official'
}

DATASET_NO_TEST = ['socialiqa', 'phys']

EMB_PATHS = {
    'transe': './data/transe/glove.transe.sgd.ent.npy',
    'lm': './data/transe/glove.transe.sgd.ent.npy',
    'numberbatch': './data/transe/concept.nb.npy',
    'tzw': './data/cpnet/tzw.ent.npy',
}

def get_node_feature_encoder(encoder_name):
    return encoder_name.replace('-cased', '-uncased')

def add_data_arguments(parser):
    # arguments that all datasets share
    parser.add_argument('--test_prediction_path', default='None', type=str)
    parser.add_argument('--save_model', default=0, type=int)

    parser.add_argument('--ent_emb', default=['transe'], choices=['transe', 'numberbatch', 'lm', 'tzw'], nargs='+', help='sources for entity embeddings')
    parser.add_argument('--ent_emb_paths', default=['./data/transe/glove.transe.sgd.ent.npy'], nargs='+', help='paths to entity embedding file(s)')
    parser.add_argument('--rel_emb_path', default='./data/transe/glove.transe.sgd.rel.npy', help='paths to relation embedding file')
    # dataset specific
    parser.add_argument('-ds', '--dataset', default='csqa', help='dataset name')
    parser.add_argument('-ih', '--inhouse', default=True, type=bool_flag, nargs='?', const=True, help='run in-house setting')
    parser.add_argument('--inhouse_train_qids', default='./data/{dataset}/inhouse_split_qids.txt', help='qids of the in-house training set')
    # statements
    parser.add_argument('--ir', default=0, type=int)
    parser.add_argument('--has_test', default=0, type=int)
    parser.add_argument('--do_pred', default=0, type=int)
    parser.add_argument('--train_statements', default='./data/{dataset}/{ir}statement/csqa_kcr/train.statement.jsonl')
    parser.add_argument('--dev_statements', default='./data/{dataset}/{ir}statement/csqa_kcr/dev.statement.jsonl')
    parser.add_argument('--test_statements', default='./data/{dataset}/{ir}statement/csqa_kcr/test.statement.jsonl')
    parser.add_argument('-ckpt', '--from_checkpoint', default='None', help='load from a checkpoint')
    # preprocessing options
    parser.add_argument('-sl', '--max_seq_len', default=64, type=int)
    # set dataset defaults
    args, _ = parser.parse_known_args()
    parser.set_defaults(ent_emb_paths=[EMB_PATHS.get(s) for s in args.ent_emb],
                        inhouse=args.inhouse,
                        inhouse_train_qids=args.inhouse_train_qids.format(dataset=args.dataset))
    data_splits = ('train', 'dev') if args.dataset in DATASET_NO_TEST else ('train', 'dev', 'test')
    for split in data_splits:
        for attribute in ('statements',):
            attr_name = f'{split}_{attribute}'
            if args.ir == 0:
                parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset, ir='')})
            else:
                parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset, ir='ir_')})
    if 'test' not in data_splits:
        parser.set_defaults(test_statements=None)


def add_encoder_arguments(parser):
    parser.add_argument('-enc', '--encoder', default='bert-large-uncased', help='encoder type')
    parser.add_argument('--encoder_name', default='roberta', help='encoder name')
    parser.add_argument('--encoder_layer', default=-1, type=int, help='encoder layer ID to use as features (used only by non-LSTM encoders)')
    parser.add_argument('-elr', '--encoder_lr', default=2e-5, type=float, help='learning rate')
    # used only for LSTM encoder
    parser.add_argument('--encoder_dim', default=128, type=int, help='number of LSTM hidden units')
    parser.add_argument('--encoder_layer_num', default=2, type=int, help='number of LSTM layers')
    parser.add_argument('--encoder_bidir', default=True, type=bool_flag, nargs='?', const=True, help='use BiLSTM')
    parser.add_argument('--encoder_dropoute', default=0.1, type=float, help='word dropout')
    parser.add_argument('--encoder_dropouti', default=0.1, type=float, help='dropout applied to embeddings')
    parser.add_argument('--encoder_dropouth', default=0.1, type=float, help='dropout applied to lstm hidden states')
    parser.add_argument('--encoder_pretrained_emb', default='./data/glove/glove.6B.300d.npy', help='path to pretrained emb in .npy format')
    parser.add_argument('--encoder_freeze_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze lstm input embedding layer')
    parser.add_argument('--encoder_pooler', default='max', choices=['max', 'mean'], help='pooling function')
    args, _ = parser.parse_known_args()
    # parser.set_defaults(encoder_lr=ENCODER_DEFAULT_LR[args.dataset].get(args.encoder, ENCODER_DEFAULT_LR['default']))


def add_optimization_arguments(parser):
    parser.add_argument('--loss', default='cross_entropy', choices=['margin_rank', 'cross_entropy'], help='model type')
    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='fixed', help='learning rate scheduler')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('--warmup_steps', type=float, default=150)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='l2 weight decay strength')
    parser.add_argument('--n_epochs', default=100, type=int, help='total number of training epochs to perform.')
    parser.add_argument('-me', '--max_epochs_before_stop', default=3, type=int, help='stop training if dev does not increase for N epochs')


def add_additional_arguments(parser):
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=False, type=bool_flag, nargs='?', const=True, help='run in debug mode')
    args, _ = parser.parse_known_args()
    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)


def get_parser():
    """A helper function that handles the arguments that all models share"""
    parser = argparse.ArgumentParser(description="Process args of prompt-QA")

    add_data_arguments(parser)
    add_encoder_arguments(parser)
    add_optimization_arguments(parser)
    add_additional_arguments(parser)

    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/', help='model output directory')
    parser.add_argument('--gen_dir', type=str)
    parser.add_argument('--gen_id', type=int)

    # for finding relation paths
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--cpnet_graph_path', default='./data/cpnet/conceptnet.en.pruned.graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    # for GPT kg generator
    parser.add_argument('--data_dir', default='csqa', type=str)
    parser.add_argument('--generator_type', default='gpt2', type=str)
    parser.add_argument('--context_len', default=16, type=int)
    parser.add_argument('--output_len', default=16, type=int, help='length of GPT2 generation')

    # data
    parser.add_argument('--train_rel_paths', default=f'./data/{args.dataset}/paths/train.relpath.2hop.jsonl')
    parser.add_argument('--dev_rel_paths', default=f'./data/{args.dataset}/paths/dev.relpath.2hop.jsonl')
    parser.add_argument('--test_rel_paths', default=f'./data/{args.dataset}/paths/test.relpath.2hop.jsonl')
    parser.add_argument('--train_adj', default=f'./data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'./data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'./data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--train_node_features',
                        default=f'./data/{args.dataset}/features/train.{get_node_feature_encoder(args.encoder)}.features.pk')
    parser.add_argument('--dev_node_features',
                        default=f'./data/{args.dataset}/features/dev.{get_node_feature_encoder(args.encoder)}.features.pk')
    parser.add_argument('--test_node_features',
                        default=f'./data/{args.dataset}/features/test.{get_node_feature_encoder(args.encoder)}.features.pk')
    parser.add_argument('--train_concepts', default=f'./data/{args.dataset}/grounded/train.grounded.jsonl')
    parser.add_argument('--dev_concepts', default=f'./data/{args.dataset}/grounded/dev.grounded.jsonl')
    parser.add_argument('--test_concepts', default=f'./data/{args.dataset}/grounded/test.grounded.jsonl')

    parser.add_argument('--node_feature_type', choices=['full', 'cls', 'mention'])
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True,
                        help='use cached data to accelerate data loading')
    parser.add_argument('--max_tuple_num', default=100, type=int)

    # model architecture
    parser.add_argument('--input_format', default="path-gen", type=str, help='input pattern template')
    parser.add_argument('--prompt_token_num', default=2, type=int, help='the number of soft prompt token')
    parser.add_argument('--ablation', default='att_pool',
                        choices=['None', 'no_kg', 'no_2hop', 'no_1hop', 'no_qa', 'no_rel',
                                 'mrloss', 'fixrel', 'fakerel', 'no_factor_mul', 'no_2hop_qa',
                                 'randomrel', 'encode_qas', 'multihead_pool', 'att_pool','no_dynamic_kg', 'no_prompt'], nargs='?', const=None,
                        help='run ablation test')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--mlp_dim', default=128, type=int, help='number of MLP hidden units')
    parser.add_argument('--mlp_layer_num', default=2, type=int, help='number of MLP layers')
    parser.add_argument('--fc_dim', default=128, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True,
                        help='freeze entity embedding layer')
    parser.add_argument('--init_range', default=0.02, type=float,
                        help='stddev when initializing with normal distribution')
    parser.add_argument('--emb_scale', default=1.0, type=float, help='scale pretrained embeddings')
    parser.add_argument('--lstm_split', default=0, type=int, help='whether to use lstm to model splited prompt tokems')
    parser.add_argument('--pattern_type', default=0, type=int, help='input pattern format')
    parser.add_argument('--pattern_format', default=None, type=str, help='input pattern format')
    parser.add_argument('--using_lstm_mlp', default=1, type=int, help='wether to use lstm and mlp to model prompt tokens')
    parser.add_argument('--using_mlp', default=0, type=int, help='wether to use lstm and mlp to model prompt tokens')
    parser.add_argument('--experiment_base', default="p-tuning", type=str, help='whether to use p-tuning based method or kcr based')
    parser.add_argument('--using_attention_for_kcr', default=0, type=int, help='whether to use attention when classification')
    parser.add_argument('--prompt_embeddings_initialized', default=0, type=int, help='whether to use hard prompt initialize')
    parser.add_argument('--concat_choices', default=0, type=int, help='whether to concat choices')
    parser.add_argument('--concat_two_choices', default=0, type=int, help='whether to concat choices')

    # regularization
    parser.add_argument('--dropoutm', type=float, default=0.3, help='dropout for mlp hidden units (0 = no dropout')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--unfreeze_epoch', default=-1, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--grad_step', default=1, type=int)
    parser.add_argument('--freeze_enc', default=1, type=int)
    parser.add_argument('--freeze_dec', default=1, type=int)


    # parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
    #                     help='show this help message and exit')

    args, _ = parser.parse_known_args()
    if args.ablation == 'mrloss':
        parser.set_defaults(loss='margin_rank')

    return parser


def get_lstm_config_from_args(args):
    lstm_config = {
        'hidden_size': args.encoder_dim,
        'output_size': args.encoder_dim,
        'num_layers': args.encoder_layer_num,
        'bidirectional': args.encoder_bidir,
        'emb_p': args.encoder_dropoute,
        'input_p': args.encoder_dropouti,
        'hidden_p': args.encoder_dropouth,
        'pretrained_emb_or_path': args.encoder_pretrained_emb,
        'freeze_emb': args.encoder_freeze_emb,
        'pool_function': args.encoder_pooler,
    }
    return lstm_config
