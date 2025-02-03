from transformers import AutoModelForCausalLM, AutoTokenizer

# 다운로드할 모델 ID (예: "MLP-KTLim/llama-3-Korean-Bllossom-8B")
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
local_model_dir = "./saved_models/llama-3-Korean-Bllossom-8B"  # 로컬 저장 경로

# 모델 및 토크나이저 다운로드
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 로컬 저장
model.save_pretrained(local_model_dir)
tokenizer.save_pretrained(local_model_dir)

