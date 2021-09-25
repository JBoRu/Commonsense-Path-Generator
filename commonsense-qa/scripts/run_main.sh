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
	--save_dir ./saved_models/csqa/f_dec_with_lstm-f_encoder_gpt-pro_tok_num_6_1-soft_prompt_GPT \
	--encoder gpt2 \
	--max_seq_len 110 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-5 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 1 \
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
	--input_format soft_prompt_p_tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_lstm_not_split_with_mlp_lr_1e4-f_enc_GPT2_medium_lr_1e4-pattern_1-prompt_tok_9-warmup_linear_700_8-bs_16-dropoutm_0.1-soft_prompt_p_tuning \
	--encoder gpt2-medium \
	--max_seq_len 115 \
	--encoder_lr 1e-4 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 4 \
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
	--freeze_dec 0 \
	--prompt_token_num 9 \
	--pattern_type 1 \
	--n_epochs 8 \
	--lr_schedule warmup_linear \
	--warmup_steps 700 \
	--lstm_split 0 \

	python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_lstm_not_split_with_mlp_lr_1e4-f_enc_GPT2_medium_lr_1e4-pattern_3-prompt_tok_3-warmup_linear_700_8-bs_16-dropoutm_0.1-soft_prompt_p_tuning \
	--encoder gpt2-medium \
	--max_seq_len 110 \
	--encoder_lr 1e-4 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 4 \
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
	--freeze_enc 0 \
	--freeze_dec 0 \
	--prompt_token_num 3 \
	--pattern_type 3 \
	--n_epochs 8 \
	--lr_schedule warmup_linear \
	--warmup_steps 700 \
	--lstm_split 0 \

	python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_lstm_not_split_with_mlp_lr_1e4-f_enc_GPT2_medium_lr_1e4-pattern_4-prompt_tok_6-warmup_linear_700_8-bs_16-dropoutm_0.1-soft_prompt_p_tuning \
	--encoder gpt2-medium \
	--max_seq_len 110 \
	--encoder_lr 1e-4 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 4 \
	--dropoutm 0.1 \
	--gpu_device 7 \
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
	--pattern_type 4 \
	--n_epochs 8 \
	--lr_schedule warmup_linear \
	--warmup_steps 700 \
	--lstm_split 0 \

	python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_lstm_not_split_with_mlp_lr_1e5-f_enc_albert_xxlarge_lr_1e5-pattern_0-prompt_tok_3-warmup_fixed-bs_16-dropoutm_0.1-soft_prompt_p_tuning \
	--encoder albert-xxlarge-v2 \
	--max_seq_len 110 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-5 \
	--batch_size 16 \
	--mini_batch_size 4 \
	--dropoutm 0.1 \
	--gpu_device 7 \
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
	--prompt_token_num 3 \
	--pattern_type 0 \
	--n_epochs 10 \
	--lr_schedule fixed \
	--lstm_split 0 \

	python main.py \
	--dataset csqa \
	--input_format p-tuning-GPT-classify \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_with_lstm-f_encoder_gpt-f_classify_head-pro_tok_num_9-p_tuning_GPT_classify \
	--encoder gpt2 \
	--max_seq_len 115 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-5 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 1 \
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
	--prompt_token_num 9 \

	python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning_classify \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_lstm_not_split_with_mlp_lr_1e4-f_enc_GPT2_medium_lr_1e4-pattern_3-prompt_tok_3-warmup_linear_700_8-bs_16-dropoutm_0.1-soft_prompt_p_tuning_classify \
	--encoder gpt2-medium \
	--max_seq_len 110 \
	--encoder_lr 1e-4 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 8 \
	--dropoutm 0.1 \
	--gpu_device 7 \
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
	--prompt_token_num 3 \
	--pattern_type 3 \
	--n_epochs 8 \
	--lr_schedule warmup_linear \
	--warmup_steps 700

	python main.py \
	--dataset csqa \
	--input_format p-tuning-GPT-classify \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_with_lstm_split-f_encoder_gpt-f_classify_head-warmup_linear-pro_tok_num_6-p_tuning_GPT_classify \
	--encoder gpt2 \
	--max_seq_len 115 \
	--encoder_lr 1e-4 \
	--decoder_lr 1e-4 \
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
	--n_epochs 7 \
	--lr_schedule warmup_linear \
	--warmup_steps 600

	python main.py \
	--dataset csqa \
	--input_format p-tuning-GPT-classify \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_without_lstm-f_encoder_gpt-f_classify_head-warmup_linear-pro_tok_num_6-p_tuning_GPT_classify \
	--encoder gpt2 \
	--max_seq_len 115 \
	--encoder_lr 1e-4 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 1 \
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
	--n_epochs 8 \
	--lr_schedule warmup_linear \
	--warmup_steps 700

	python main.py \
	--dataset csqa \
	--input_format p-tuning-GPT-classify \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_without_lstm-f_encoder_gpt-f_classify_head-warmup_linear-pro_tok_num_9-p_tuning_GPT_classify \
	--encoder gpt2 \
	--max_seq_len 115 \
	--encoder_lr 1e-4 \
	--decoder_lr 1e-4 \
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
	--prompt_token_num 9 \
	--n_epochs 15 \
	--lr_schedule warmup_linear \
	--warmup_steps 1000

	python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_lstm_not_split_with_mlp_lr_1e4-f_enc_GPT2_lr_1e4-pattern_3-prompt_tok_3-warmup_linear_700_8-bs_16-dropoutm_0.1-soft_prompt_p_tuning \
	--encoder gpt2 \
	--max_seq_len 110 \
	--encoder_lr 1e-4 \
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
	--freeze_enc 0 \
	--freeze_dec 0 \
	--pattern_type 3 \
	--prompt_token_num 3 \
	--n_epochs 8 \
	--lr_schedule warmup_linear \
	--warmup_steps 700 \
	--lstm_split 0 \

	python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning_classify \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_lstm_not_split_with_mlp_lr_1e4-f_enc_roberta_large_lr_1e4-pattern_0-prompt_tok_3-warmup_linear_450_8-bs_16-dropoutm_0.1-soft_prompt_p_tuning_classify \
	--encoder roberta-large \
	--max_seq_len 100 \
	--encoder_lr 1e-4 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 2 \
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
	--prompt_token_num 3 \
	--n_epochs 8 \
	--lr_schedule warmup_linear \
	--warmup_steps 450 \
	--lstm_split 0 \

	python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning_classify \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_lstm_not_split_with_mlp_lr_1e5-f_enc_albert_xxlarge_v2_lr_1e5-pattern_4-prompt_tok_3-warmup_fixed-bs_16-dropoutm_0.1-soft_prompt_p_tuning_classify \
	--encoder albert-xxlarge-v2 \
	--max_seq_len 100 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-5 \
	--batch_size 16 \
	--mini_batch_size 4 \
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
	--pattern_type 4 \
	--prompt_token_num 3 \
	--n_epochs 8 \
	--lr_schedule fixed \
	--lstm_split 0 \

	python main.py \
	--dataset csqa \
	--input_format p-tuning-GPT-generate \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_with_lstm_not_split-f_encoder_gpt-f_classify_head-warmup_linear-pro_tok_num_6-p-tuning-GPT-generate \
	--encoder gpt2 \
	--max_seq_len 115 \
	--encoder_lr 1e-4 \
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
	--ablation no_dynamic_kg \
	--freeze_enc 0 \
	--freeze_dec 0 \
	--prompt_token_num 6 \
	--n_epochs 8 \
	--lr_schedule warmup_linear \
	--warmup_steps 700


python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_lstm_not_split_with_mlp_lr_1e5-f_enc_Roberta_large-pattern_2-prompt_tok_6-warmup_fixed-bs_16-dropoutm_0.1-soft_prompt_p_tuning \
	--encoder roberta-large \
	--max_seq_len 110 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-5 \
	--batch_size 16 \
	--mini_batch_size 8 \
	--dropoutm 0.1 \
	--gpu_device 0 \
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
	--prompt_token_num 6 \
	--pattern_type 2 \
	--n_epochs 20 \
	--lr_schedule fixed \
	--lstm_split 0 \

	python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_only_mlp_lr_1e4-uf_enc_roberta_large-pattern_0-prompt_tok_100-warmup_fixed-bs_16-dropoutm_0.1-soft_prompt_p_tuning \
	--encoder roberta-large \
	--max_seq_len 200 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 1 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 2 \
	--fc_dim  2048 \
	--fc_layer_num 1 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 1 \
	--freeze_dec 0 \
	--prompt_token_num 100 \
	--pattern_type 0 \
	--n_epochs 20 \
	--lr_schedule fixed \
	--using_lstm_mlp 0 \
	--using_mlp 1 \

	python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_only_mlp_lr_1e4-uf_enc_roberta_large-pattern_3-prompt_tok_100-warmup_fixed-bs_16-dropoutm_0.1-soft_prompt_p_tuning \
	--encoder roberta-large \
	--max_seq_len 200 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 16 \
	--dropoutm 0.1 \
	--gpu_device 1 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 2 \
	--fc_dim  2048 \
	--fc_layer_num 1 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 1 \
	--freeze_dec 0 \
	--prompt_token_num 100 \
	--pattern_type 3 \
	--n_epochs 20 \
	--lr_schedule fixed \
	--using_lstm_mlp 0 \
	--using_mlp 1 \

python main.py \
	--dataset csqa \
	--input_format soft_prompt_p_tuning \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_dec_only_mlp_lr_1e4-uf_enc_albert_xxlarge_v2-pattern_0-prompt_tok_100-warmup_fixed-bs_16-dropoutm_0.1-soft_prompt_p_tuning \
	--encoder albert-xxlarge-v2 \
	--max_seq_len 200 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 8 \
	--dropoutm 0.1 \
	--gpu_device 2 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 2 \
	--fc_dim  2048 \
	--fc_layer_num 1 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 1 \
	--freeze_dec 0 \
	--prompt_token_num 100 \
	--pattern_type 0 \
	--n_epochs 20 \
	--lr_schedule fixed \
	--using_lstm_mlp 0 \
	--using_mlp 1 \

python main.py \
	--dataset csqa \
	--input_format manual_hard_prompt \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_enc_albert_xxlarge_lr_1e5-pattern_0-warmup_fixed-bs_16-dropoutm_0.1-manual_hard_prompt \
	--encoder albert-xxlarge-v2 \
	--max_seq_len 100 \
	--encoder_lr 1e-5 \
	--decoder_lr 1e-4 \
	--batch_size 16 \
	--mini_batch_size 4 \
	--dropoutm 0.1 \
	--gpu_device 3 \
	--nprocs 20 \
	--save_model 1 \
	--mlp_dim 300 \
	--mlp_layer_num 2 \
	--fc_dim  2048 \
	--fc_layer_num 1 \
	--seed 42 \
	--ablation no_dynamic_kg \
	--freeze_enc 0 \
	--freeze_dec 1 \
	--prompt_token_num 100 \
	--pattern_type 0 \
	--n_epochs 20 \
	--lr_schedule fixed \
	--lstm_split 0 \

python main.py \
	--dataset csqa \
	--input_format manual_hard_prompt \
	--inhouse 1 \
	--save_dir ./saved_models/csqa/f_enc_roberta_large_lr_1e5-pattern_0-warmup_fixed-bs_16-dropoutm_0.1-manual_hard_prompt \
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
	--freeze_dec 1 \
	--prompt_token_num 100 \
	--pattern_type 0 \
	--n_epochs 20 \
	--lr_schedule fixed \
	--lstm_split 0 \
