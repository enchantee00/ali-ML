from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import torch
import faiss
import pickle
import json
import asyncio
import gc
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
import uvicorn
import time  # 추가
from llm import *

app = FastAPI()

model_path = "nlpai-lab/KURE-v1"
model_kwargs = {"device": "cuda:1"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

sentence_model = SentenceTransformer(model_path)
llm = setup_llm_pipeline()
conversation_memory = {}

FAISS_DB_PATHS = {
    "고용절차": "./data/faiss/kure-v1/고용절차/고용절차",
    "취업 및 준수사항": "./data/faiss/kure-v1/취업 및 준수사항/취업 및 준수사항",
    "노동법 및 권리보호": "./data/faiss/kure-v1/노동법 및 권리보호/노동법 및 권리보호",
    "출입국 및 체류": "./data/faiss/kure-v1/출입국 및 체류/출입국 및 체류",
    "사용자 의무": "./data/faiss/kure-v1/사용자 의무/사용자 의무"
}

CATEGORY_DESCRIPTIONS = {
    "고용절차": "외국인 근로자의 고용 과정과 절차, 고용 자격, 고용 제한 및 허가와 관련된 정보",
    "취업 및 준수사항": "외국인 근로자의 취업 절차, 자격 요건, 보험 가입, 출입국 관리법 등의 준수사항",
    "노동법 및 권리보호": "외국인 근로자의 노동법 관련 보호, 근로 기준법, 최저임금 및 권리 구제 절차",
    "출입국 및 체류": "외국인의 출입국 절차, 체류 관련 법률, 비자, 국제결혼 및 장례 관련 정보",
    "사용자 의무": "고용주의 법적 의무, 고용 변동 신고, 근로 조건, 보험 가입 및 성희롱 예방 조치"
}

category_sentences = list(CATEGORY_DESCRIPTIONS.values())
category_names = list(CATEGORY_DESCRIPTIONS.keys())
category_embeddings = sentence_model.encode(category_sentences, normalize_embeddings=True)

embedding_size = category_embeddings.shape[1]
category_index = faiss.IndexFlatL2(embedding_size)
category_index.add(category_embeddings)  # 카테고리 벡터 추가


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


def determine_category(question: str) -> str:
    """KURE-v1을 사용하여 질문을 임베딩 후 FAISS에서 가장 가까운 카테고리를 선택"""
    query_embedding = sentence_model.encode([question], normalize_embeddings=True)
    D, I = category_index.search(query_embedding, k=1)  # 가장 가까운 카테고리 검색
    return category_names[I[0][0]]  # 가장 가까운 카테고리 반환


def load_faiss_db(faiss_db_directory: str):
    """FAISS DB를 로드하는 함수"""
    with open(faiss_db_directory + "_index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)

    with open(faiss_db_directory + "_docstore.pkl", "rb") as f:
        docstore = pickle.load(f)

    index = faiss.read_index(faiss_db_directory + "_faiss_db.index")

    db = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3, "fetch_k": 8})
    return db, retriever


def get_conversation_memory(session_id: str):
    """세션 ID 기반으로 대화 메모리를 반환 (없으면 생성)"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = ConversationBufferMemory()
    return conversation_memory[session_id]


@app.get("/")
def home():
    return {"message": "Hello, World!"}


@app.post("/ask")
async def ask(request: QuestionRequest):
    start_time = time.time()  # 시작 시간 기록

    question = request.question
    session_id = request.session_id

    # 세션 ID 기반 대화 메모리 가져오기
    memory = get_conversation_memory(session_id)
    memory.save_context({"input": question}, {"output": ""})

    # 질문에서 카테고리 판별
    category = determine_category(question)

    if category not in FAISS_DB_PATHS:
        raise HTTPException(status_code=400, detail="No matching category found")

    # 해당 카테고리의 FAISS DB 로드
    db_path = FAISS_DB_PATHS[category]
    db, retriever = load_faiss_db(db_path)

    # RAG 실행 (이전 대화 내용과 함께)
    rag_chain = rag(retriever, llm)

    with torch.no_grad():  # 메모리 최적화
        # 이전 대화 히스토리를 함께 전달하여 LLM 호출
        full_prompt = memory.load_memory_variables({})["history"] + "\n" + question
        response = await asyncio.to_thread(rag_chain.invoke, full_prompt)

    # 답변을 대화 메모리에 저장 (맥락 유지)
    memory.save_context({"input": question}, {"output": response})

    # 처리 시간 계산
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)  
    print(f"processing_time: {processing_time}")

    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "category": category,
        "answer": response,
        "session_id": session_id
    }


# FastAPI 실행 (Uvicorn)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
