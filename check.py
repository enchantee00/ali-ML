import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "rtzr/ko-gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델 로드 (자동 설정)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 확인 1️⃣: Config에서 직접 가져오기
attn_implementation = model.config.__dict__.get("attn_implementation", "자동 설정됨 (_attn_implementation_autoset=True)")
print(f"현재 Attention 설정: {attn_implementation}")

# 확인 2️⃣: 모델을 실제로 실행하여 Debugging
input_text = "이 모델의 attention 구현 방식은 무엇인가요?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

with torch.no_grad():
    output = model(input_ids)

# 확인 3️⃣: SDPA 또는 EAGER 사용 여부 확인
if hasattr(model, "attn_implementation"):
    print(f"🔹 모델이 실제 사용하는 Attention 구현 방식: {model.attn_implementation}")
else:
    print("⚠️ 모델이 자동 설정된 Attention을 사용 중 (SDPA일 가능성 높음)")
