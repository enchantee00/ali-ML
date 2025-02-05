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
    # config.attn_implementation = "flash_attention_2"  # eager 방식 사용

    # 모델 로드 및 양자화 설정 적용
    model = Gemma2ForCausalLM.from_pretrained(
        model_id,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    # model = AutoModelForCausalLM.from_pretrained("./models/saved_models/llama3-8b", device_map="auto", quantization_config=bnb_config) 
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # LoRA 어댑터 적용 (선택 사항)
    # lora_adapter_path = "./models/lora_adapter/gemma2"
    # if os.path.exists(lora_adapter_path):
    #     model = PeftModel.from_pretrained(model, lora_adapter_path)

    # model.generation_config.cache_implementation = "static"
    # model = torch.compile(model, mode="max-autotune")  #

    # HuggingFacePipeline 객체 생성
    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False, # True -> 입력 프롬프트 + 생성된 답변
        device_map="auto",
        repetition_penalty=1.2,
        # eos_token_id=tokenizer.eos_token_id, # 완전한 문장 생성 유도
        max_new_tokens=512,
        do_sample=False,  # 확률 기반 샘플링 없이 가장 확률 높은 단어 선택
        temperature=0.5
    )

    hf_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return tokenizer, model, hf_pipeline


def normalize_string(s):
    """한글 문자열 정규화"""
    return unicodedata.normalize('NFC', s)


def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷팅"""
    return "\n".join(doc.page_content for doc in docs)


def rag_llama(llm):
    template = """<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>

    당신은 법률 전문가 AI입니다. 주어진 법률 문서를 참고하여 사용자의 질문에 대해 정확하고 신뢰할 수 있는 답변을 제공하세요.
    답변 시 다음 기준을 따르세요:
    - 질문과 관련된 법률 조항만 요약하여 답변하세요.
    - 답변은 명확하고 완결된 문장으로 구성하세요.
    - 동일한 정보를 반복하지 마세요.

    <|start_header_id|>user<|end_header_id|>

    ### 이전 대화:
    {history_text}

    ### 참고 법률 문서:
    {context}

    ### 질문:
    {question}

    <|start_header_id|>assistant<|end_header_id|>
    """

    prompt = PromptTemplate.from_template(template)

    # RAG 체인 정의
    rag_chain = (
        {
            "history_text": RunnablePassthrough(), 
            "context": RunnablePassthrough(),       
            "question": RunnablePassthrough()      
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def rag_gemma(llm):
    template = """<bos><start_of_turn>system
    당신은 법률 전문가 AI입니다. 참고 문서를 바탕으로 정확하고 논리적인 답변을 제공하세요.
    - 질문과 관련된 법률 조항만 요약하여 답변하세요.
    - 답변은 명확하고 완결된 문장으로 구성하세요.
    - 동일한 정보를 반복하지 마세요.

    <end_of_turn>
    <start_of_turn>user
    이전 대화 내용:
    {history_text}

    참고 문서 내용:
    {context}

    질문:
    {question}

    명확하고 논리적인 문장으로 답변하세요. 질문에 대한 핵심 개념을 포함하여 일관된 문장으로 설명하세요.
    <end_of_turn>
    <start_of_turn>model
    """

    prompt = PromptTemplate.from_template(template)

    # RAG 체인 정의
    rag_chain = (
        {
            "history_text": RunnablePassthrough(), 
            "context": RunnablePassthrough(),       
            "question": RunnablePassthrough()      
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain