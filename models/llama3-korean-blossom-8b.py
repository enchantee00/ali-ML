from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
)
from datasets import Dataset, load_from_disk

import os, torch, json, wandb, subprocess
from sklearn.model_selection import train_test_split
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
)
import torch.nn as nn
from trl import SFTTrainer, SFTConfig

torch.cuda.empty_cache()  
torch.cuda.ipc_collect()  


def generate_prompts(examples):
    prompt_list = []
    for context, question, answer in zip(examples["context"], examples["question"], examples["answer"]):
        prompt_list.append(
            f"""<|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>

            당신은 법률 전문가 AI입니다. 주어진 법률 문서를 참고하여 사용자의 질문에 대해 정확하고 신뢰할 수 있는 답변을 제공하세요.
            법률 용어를 명확하게 사용하고, 신뢰할 수 있는 정보를 바탕으로 근거를 제시하세요.

            <|start_header_id|>user<|end_header_id|>

            ### 법률 문서:
            {context}

            ### 질문:
            {question}

            ### 답변 지침:
            - 법률 조항 또는 관련 판례를 근거로 하여 답변하세요.
            - 사용자가 이해하기 쉽도록 간결하고 논리적으로 설명하세요.
            - 필요 시, 법 조항의 원문을 인용하세요.

            <|start_header_id|>assistant<|end_header_id|>

            {answer}<|eot|>"""
        )
    return prompt_list




data_save_path = "../data/processed"
train_dataset = load_from_disk(os.path.join(data_save_path, "train"))
val_dataset = load_from_disk(os.path.join(data_save_path, "val"))


model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16) 
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = model.to_empty(device="cuda:0")


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)


# LoRA 모델 생성
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# model.gradient_checkpointing_enable()
model.train()


# 훈련 인자 설정
training_args = SFTConfig(
    output_dir="../results/llama3",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    eval_strategy="steps",
    eval_steps=0.1,
    logging_dir="./logs",
    logging_steps=11,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    group_by_length=True,
    bf16=True,
    report_to="wandb",
    run_name="llama-3-8b-lora-bf16-0202",
    max_seq_length = 512
)
 
# Trainer 초기화 및 훈련
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    args=training_args,
    formatting_func=generate_prompts,
)

model.config.use_cache = False
trainer.train()

adapter_save_path = "./lora_adapterl/llama3"
trainer.save_model(adapter_save_path)
print(f"LoRA 어댑터가 {adapter_save_path}에 저장되었습니다!")