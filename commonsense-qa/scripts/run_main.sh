#python main.py \
#	--dataset csqa \
#	--input_format p-tuning \
#	--inhouse 1 \
#	--save_dir ./saved_models/csqa/dlr_1e-4_only-mlp_kg-mlp-2_merge-mlp-1 \
#	--encoder roberta-large \
#	--max_seq_len 100 \
#	--encoder_lr 2e-6 \
#	--decoder_lr 1e-4 \
#	--batch_size 16 \
#	--mini_batch_size 16 \
#	--dropoutm 0.1 \
#	--gpu_device 0 \
#	--nprocs 20 \
#	--save_model 1 \
#	--mlp_dim 300 \
#	--mlp_layer_num 2 \
#	--fc_dim  4096 \
#	--fc_layer_num 1 \
#	--seed 42 \
python main.py \
	--dataset csqa \
	--input_format p-tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/dlr_1e_4-unfreeze_encoder-unfreeze_embedding-mlp_kg_mlp_2-merge_mlp_1 \
	--encoder roberta-large \
	--max_seq_len 100 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 8 \
	--dropoutm 0.1 \
	--gpu_device 3 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 2 \
	--fc_dim  2048 \
	--fc_layer_num 1 \
	--seed 42 \
#python main.py \
#	--dataset csqa \
#	--input_format p-tuning \
#	--inhouse 1 \
#	--save_dir ./saved_models/csqa/dlr_1e-4_unfreeze_embedding_no_dyn_kg_mlp_kg-mlp-2_merge-mlp-1 \
#	--encoder roberta-large \
#	--max_seq_len 100 \
#	--encoder_lr 2e-6 \
#	--decoder_lr 1e-4 \
#	--batch_size 16 \
#	--mini_batch_size 16 \
#	--dropoutm 0.1 \
#	--gpu_device 3 \
#	--nprocs 20 \
#	--save_model 1 \
#	--mlp_dim 300 \
#	--mlp_layer_num 2 \
#	--fc_dim  2048 \
#	--fc_layer_num 1 \
#	--seed 42 \
#	--ablation no_dynamic_kg \
#python main.py \
#	--dataset csqa \
#	--input_format p-tuning \
#	--inhouse 1 \
#	--save_dir ./saved_models/csqa/dlr_1e-4_unfreeze-encoder_unfreeze-embedding_no-dyn-kg_mlp_kg-mlp-2_merge-mlp-1 \
#	--encoder roberta-large \
#	--max_seq_len 100 \
#	--encoder_lr 2e-6 \
#	--decoder_lr 1e-4 \
#	--batch_size 16 \
#	--mini_batch_size 16 \
#	--dropoutm 0.1 \
#	--gpu_device 2 \
#	--nprocs 20 \
#	--save_model 1 \
#	--mlp_dim 300 \
#	--mlp_layer_num 2 \
#	--fc_dim  2048 \
#	--fc_layer_num 1 \
#	--seed 42 \
#	--ablation no_dynamic_kg \
python main.py \
	--dataset csqa \
	--input_format no-prompt \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/elr_1e-5_no_prompt \
	--encoder roberta-large \
	--max_seq_len 100 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 5 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 2 \
	--fc_dim  2048 \
	--fc_layer_num 1 \
	--seed 42 \
	--ablation no_dynamic_kg \
