from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 토크나이저와 모델 불러오기 (device_map="auto"로 자동 할당)
tokenizer = AutoTokenizer.from_pretrained("DopeorNope/Ko-Mixtral-v1.4-MoE-7Bx2")
model = AutoModelForCausalLM.from_pretrained("DopeorNope/Ko-Mixtral-v1.4-MoE-7Bx2", device_map="auto")

text = '지능(智能) 또는 인텔리전스(intelligence)는 인간의 <MASK> 능력을 말한다.'
input_ids = tokenizer.encode(text, return_tensors="pt")

# 모델이 이미 적절한 디바이스에 배치되어 있으므로 바로 생성 가능
output_ids = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_k=50,
    top_p=0.95
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated text:")
print(output_text)
