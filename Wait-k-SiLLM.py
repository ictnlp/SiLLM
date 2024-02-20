import os
import sys
import pdb
import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from datasets import load_dataset
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import json

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    data_path: str = "",
    output_translation_path: str="",
    waitk: int=1,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        output=None,
        suppress_tokens=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input, output)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            num_beams=num_beams,
            suppress_tokens=suppress_tokens,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output), s.size(-1) - input_ids.size(-1)

    def Waitk_policy(
        instruction,
        input=None,
        num_beams=1,
        waitk=1,
        max_new_tokens=256
    ):
        cur_target_str = ""
        tokenized_input = input
        i = 0
        src_len = len(input.split())
        tmp_max_new_tokens = 1
        rw_seq = []
        first_time = True
        suppress_tokens=[2]
        while (i+waitk <= src_len) or first_time:
            cut_input = ' '.join(input.split()[:min(i+waitk, src_len)])
            tmp_max_new_tokens = 5
            if i+waitk >= src_len:
                tmp_max_new_tokens = max_new_tokens
                suppress_tokens=None
            cur_target_str, tmp_size = evaluate(instruction, cut_input, output=cur_target_str, suppress_tokens=suppress_tokens, num_beams=num_beams, max_new_tokens=tmp_max_new_tokens)
            if i+waitk < src_len:
                cur_target_str = ' '.join(cur_target_str.split()[:i+1])
                rw_seq.append(i+waitk)
                if cur_target_str.find('</s>') != -1:
                    break
            else:
                tmp_size = len(cur_target_str.split()) - i
                rw_seq = rw_seq + [src_len] * tmp_size
            first_time=False
            i += 1
        rw_seq.append(src_len)
        
        return rw_seq, cur_target_str
    data = load_dataset("json", data_files=data_path)
    test_data = data["train"]
    output_text = []
    j = 1
    for item_data in test_data:
        print('sample' + str(j))
        j += 1
        tmp_result = Waitk_policy(item_data["instruction"], item_data["input"], num_beams=1, waitk=waitk, max_new_tokens=1024)
        index = tmp_result[1].find('\n')
        tmp_str = tmp_result[1]
        if index!=-1:
            tmp_str = tmp_result[1][:index]
        output_text.append({'rw': tmp_result[0], 'translation': tmp_str})
    with open(output_translation_path, "w", encoding='utf-8') as fp:
        json.dump(output_text, fp, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    fire.Fire(main)

