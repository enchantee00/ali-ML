import os
import unicodedata

import torch
import pandas as pd
from tqdm import tqdm
import fitz  # PyMuPDF

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    Gemma2ForCausalLM
)
from accelerate import Accelerator

# Langchain 관련
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore import InMemoryDocstore


from peft import PeftModel
import faiss


def setup_llm_pipeline():
    torch.cuda.empty_cache()
    
    # 4비트 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 모델 ID 
    model_id = "rtzr/ko-gemma-2-9b-it"

    # 토크나이저 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False
    

    # 모델 로드 및 양자화 설정 적용
    model = Gemma2ForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True )

#     model = PeftModel.from_pretrained(model, "./persona/checkpoint-200",is_trainable=True)

    # HuggingFacePipeline 객체 생성
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        #temperature=0.2,
        return_full_text=False,
        max_new_tokens=450,
    )

    hf = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return hf


def normalize_string(s):
    return unicodedata.normalize('NFC', s)


def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷팅"""
    context = ""
    for doc in docs:
        context += doc.page_content
        context += '\n'
    return context

def rag(retriever, llm):
    # RAG 체인 구성
    template = """
    다음 정보를 바탕으로 질문에 답하세요:
    {context}

    질문: {question}

    주어진 질문에만 답변하세요. 문장으로 답변해주세요. 답변할 때 질문의 주어를 써주세요.
    답변:
    """
    prompt = PromptTemplate.from_template(template)

    # RAG 체인 정의
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain