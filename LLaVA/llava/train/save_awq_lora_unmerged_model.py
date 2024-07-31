from awq import AutoAWQForCausalLM
from llava.model.builder import load_pretrained_model
import os
import torch
def save_full_awq_model():
    model_path = "./checkpoints/llava-v1.6-mistral-7b-orpo-lora"
    model_name = None
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    if model_name is None:
        model_paths = model_path.split("/")
        if model_paths[-1].startswith('checkpoint-'):
            model_name = model_paths[-2] + "_" + model_paths[-1]
        else:
            model_name = model_paths[-1]
    else:
        model_name = model_name
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, "liuhaotian/llava-v1.6-mistral-7b", model_name, load_8bit=False, load_4bit=False, device="cuda", use_flash_attn=False)

    # Save the merged full-sized model
    merged_output_dir = os.path.join(model_path, 'merged_model')
    os.makedirs(merged_output_dir, exist_ok=True)
    #model.save_pretrained(merged_output_dir)
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
    print("Saving AWQ model...")
    model.config.architectures = "MistralForCausalLM"
    # Quantize the merged model with AWQ
    awq_model = AutoAWQForCausalLM.from_pretrained(merged_output_dir, torch_dtype=torch.float16)
    awq_model.quantize(tokenizer,quant_config)  # You can adjust the number of bits as needed
    awq_output_dir = os.path.join(model_path, 'awq')
    awq_model.save_quantized(awq_output_dir)
    tokenizer.save_pretrained(awq_output_dir)
    
if __name__ == "__main__":
    save_full_awq_model()