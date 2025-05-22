import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

peft_model_path = "./tinyllama_finetuned"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Lấy config từ mô hình LoRA
config = PeftConfig.from_pretrained(peft_model_path)

# Load mô hình gốc với 4bit
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=quantization_config,
    device_map="auto"
)

# Áp dụng LoRA
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Input
prompt = "Giải thích quy trình xử lý đơn hàng."

# Token hóa đầu vào
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Sinh văn bản
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# In kết quả
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
