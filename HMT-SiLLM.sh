Base_Model=/path/base_model
LoRA_Weithts=/path/LoRA_weights
Output_Translation=/path/output
Test_Data=./HMT_Policy/L2_K4.json
Bottom=1
Top=3

python HMT-SiLLM.py \
    --base_model ${Base_Model} \
    --lora_weights ${LoRA_Weithts} \
    --prompt_template 'Text_translation' \
    --Bottom ${Bottom} \
    --Top ${Top} \
    --data_path ${Test_Data} \
    --output_translation_path ${Output_Translation}
