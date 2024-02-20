k=11
Base_Model=/path/base_model
LoRA_Weithts=/path/LoRA_weights
Output_Translation=/path/output
Test_Data=./test.json

python Wait-k-SiLLM.py \
    --base_model ${Base_Model} \
    --lora_weights ${LoRA_Weithts} \
    --prompt_template 'Text_translation' \
    --data_path ${Test_Data} \
    --output_translation_path ${Output_Translation} \
    --waitk ${k}
