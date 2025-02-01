import os
import unicodedata
import torch
import fitz  # PyMuPDF
import faiss
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    Gemma2ForCausalLM,
    Gemma2Config
)
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


def setup_llm_pipeline():
    """LoRA 적용 및 4비트 양자화 모델 설정"""
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

    config = Gemma2Config.from_pretrained(model_id)
    config.attn_implementation = "eager"  # eager 방식 사용

    # 모델 로드 및 양자화 설정 적용
    model = Gemma2ForCausalLM.from_pretrained(
        model_id,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA 어댑터 적용 (선택 사항)
    lora_adapter_path = "./models/lora_adapter"
    if os.path.exists(lora_adapter_path):
        model = PeftModel.from_pretrained(model, lora_adapter_path)

    # HuggingFacePipeline 객체 생성
    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=450,
        device_map="auto"
    )

    hf_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return hf_pipeline


def normalize_string(s):
    """한글 문자열 정규화"""
    return unicodedata.normalize('NFC', s)


def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷팅"""
    return "\n".join(doc.page_content for doc in docs)


def rag(retriever, llm):
    template = """
    <bos><start_of_turn>user
    다음 정보를 바탕으로 질문에 답하세요.

    Context: {context}

    Question: {question}

    주어진 질문에만 답변하세요. 문장으로 답변해주세요. 답변할 때 질문의 주어를 써주세요.
    <end_of_turn>
    <start_of_turn>model
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
