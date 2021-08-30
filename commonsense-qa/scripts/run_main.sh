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
#	--fc_dim  2048 \
#	--fc_layer_num 1 \
#	--seed 42 \
#	> ./saved_models/csqa/dlr_1e-4_only-mlp_kg-mlp-2_merge-mlp-1/train.log 2>&1
#python main.py \
#	--dataset csqa \
#	--input_format p-tuning \
#	--inhouse 1 \
#	--save_dir ./saved_models/csqa/dlr_1e_4-unfreeze_encoder-unfreeze_embedding-mlp_kg_mlp_2-merge_mlp_1 \
#	--encoder roberta-large \
#	--max_seq_len 100 \
#	--encoder_lr 1e-5 \
#	--decoder_lr 1e-4 \
#	--batch_size 16 \
#	--mini_batch_size 8 \
#	--dropoutm 0.1 \
#	--gpu_device 3 \
#	--nprocs 20 \
#	--save_model 1 \
#	--mlp_dim 300 \
#	--mlp_layer_num 2 \
#	--fc_dim  2048 \
#	--fc_layer_num 1 \
#	--seed 42 \
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
	--input_format soft-prompt \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/dlr_1e_4-f_enc-uf_dec-soft_prompt \
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
	--freeze_enc True \
	--freeze_dec False \
	--prompt_token_num 3 \

python main.py \
	--dataset csqa \
	--input_format soft-prompt \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/elr_1e_5-dlr_1e_4-uf_enc-uf_dec-soft_prompt \
	--encoder roberta-large \
	--max_seq_len 100 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 8 \
	--dropoutm 0.1 \
	--gpu_device 4 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 2 \
	--fc_dim  2048 \
	--fc_layer_num 1 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 0 \
	--freeze_dec 0 \
	--prompt_token_num 3 \

nohup python main.py \
	--dataset csqa \
	--input_format p-tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/dlr_1e_4-only_mlp-pro_tok_num_10-p_tuning \
	--encoder roberta-large \
	--max_seq_len 110 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 4 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 1 \
	--fc_dim  2048 \
	--fc_layer_num 1 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 1 \
	--freeze_dec 1 \
	--prompt_token_num 10 \
	> ./saved_models/csqa/dlr_1e_4-only_mlp-pro_tok_num_10-p_tuning_train.log 2>&1 &

python main.py \
	--dataset csqa \
	--input_format p-tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/dlr_1e_4-only_mlp-with_dyn_kg-pro_tok_num_10-p_tuning \
	--encoder roberta-large \
	--max_seq_len 110 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 5 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 1 \
	--fc_dim  2048 \
	--fc_layer_num 2 \
	--seed 42 \
	--ablation None \
	--freeze_enc 1 \
	--freeze_dec 1 \
	--prompt_token_num 10 \

python main.py \
	--dataset csqa \
	--input_format p-tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/dlr_1e_4-mlp_with_kg_embedding-pro_tok_num_10-p_tuning \
	--encoder roberta-large \
	--max_seq_len 110 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 6 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 1 \
	--fc_dim  2048 \
	--fc_layer_num 2 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 1 \
	--freeze_dec 0 \
	--prompt_token_num 10 \

	python main.py \
	--dataset csqa \
	--input_format p-tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/elr_1e_5-dlr_1e_4-uf_encoder-mlp-pro_tok_num_10-p_tuning \
	--encoder roberta-large \
	--max_seq_len 110 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 8 \
	--dropoutm 0.1 \
	--gpu_device 5 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 1 \
	--fc_dim  2048 \
	--fc_layer_num 2 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 0 \
	--freeze_dec 1 \
	--prompt_token_num 10 \

	python main.py \
	--dataset csqa \
	--input_format p-tuning-GPT \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/dlr_1e_5-uf_encoder_gpt-pro_tok_num_32-p_tuning_GPT \
	--encoder roberta-large \
	--max_seq_len 130 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-5 \
	--batch_size 16 \
	--mini_batch_size 8 \
	--dropoutm 0.1 \
	--gpu_device 6 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 1 \
	--fc_dim  2048 \
	--fc_layer_num 2 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 1 \
	--freeze_dec 0 \
	--prompt_token_num 36 \

	python main.py \
	--dataset csqa \
	--input_format soft_prompt \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/uf_dec-f_encoder_gpt-pro_tok_num_6-soft_prompt_GPT \
	--encoder gpt2 \
	--max_seq_len 110 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-5 \
	--batch_size 16 \
	--mini_batch_size 8 \
	--dropoutm 0.1 \
	--gpu_device 3 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 1 \
	--fc_dim  2048 \
	--fc_layer_num 2 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 1 \
	--freeze_dec 0 \
	--prompt_token_num 6 \

	python main.py \
	--dataset csqa \
	--input_format soft-prompt \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_with_lstm-f_encoder_gpt-pro_tok_num_6-soft_prompt_GPT \
	--encoder gpt2 \
	--max_seq_len 110 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-5 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 0 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 1 \
	--fc_dim  2048 \
	--fc_layer_num 2 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 0 \
	--freeze_dec 0 \
	--prompt_token_num 6 \

	python main.py \
	--dataset csqa \
	--input_format soft-prompt \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_with_mlp-f_encoder_gpt-pro_tok_num_6-soft_prompt_GPT \
	--encoder gpt2 \
	--max_seq_len 110 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-5 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 3 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 1 \
	--fc_dim  2048 \
	--fc_layer_num 2 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 0 \
	--freeze_dec 0 \
	--prompt_token_num 6 \
	--n_epochs 10 \
	--lr_schedule warmup_linear \
	--warmup_steps 900