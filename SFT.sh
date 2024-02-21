Base_Model=/path/base_model
LoRA_Weithts=/path/LoRA_weights
Data_File=./SFT_Data

python finetune.py \
 --base_model  ${Base_Model} \
 --data_path ${Data_File} \
 --output_dir ${LoRA_Weithts} \
 --batch_size 128 \
 --micro_batch_size 4 \
 --num_epochs 10 \
 --learning_rate 1e-4 \
 --cutoff_len 1024 \
 --val_set_size 2000 \
 --lora_r 8 \
 --cutoff_len 1024 \
 --lora_alpha 16 \
 --lora_dropout 0.05 \
 --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
 --train_on_inputs \
 --group_by_length \
 --train_on_inputs False \
 --prompt_template_name 'Text_translation'
