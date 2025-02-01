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
    prompt_list=[]
    for context, question, answer in zip(examples["context"], examples["question"], examples["answer"]):
        prompt_list.append(
            f"""<bos><start_of_turn>user
            다음 문서를 참고하여 질문에 답변해주세요:
            
            Context: {context}
            Question: {question}
            <end_of_turn>
            <start_of_turn>model
            {answer}<end_of_turn><eos>"""
        )
    return prompt_list


data_save_path = "../data/processed"
train_dataset = load_from_disk(os.path.join(data_save_path, "train"))
val_dataset = load_from_disk(os.path.join(data_save_path, "val"))


model_name = "rtzr/ko-gemma-2-9b-it"
model = AutoModelForCausalLM.from_pretrained("./saved_models/ko-gemma-2-9b-it", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager") 
tokenizer = AutoTokenizer.from_pretrained(model_name)


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "gate_proj"],
)


# LoRA 모델 생성
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# model.gradient_checkpointing_enable()
model.train()


# 훈련 인자 설정
training_args = SFTConfig(
    output_dir="../results",
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
    run_name="gemma-2-9b-lora-bf16-0926",
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

adapter_save_path = "./lora_adapter"
trainer.save_model(adapter_save_path)
print(f"LoRA 어댑터가 {adapter_save_path}에 저장되었습니다!")