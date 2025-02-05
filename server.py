from fastapi import FastAPI, HTTPException, Depends, Header
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
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
import uvicorn
import time 
from torch.nn.attention import SDPBackend, sdpa_kernel

from llm import *

app = FastAPI()

API_KEY = "RIS-alitouch-key"


model_path = "nlpai-lab/KURE-v1"
model_kwargs = {"device": "cuda:1"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

sentence_model = SentenceTransformer(model_path)
sentence_model = torch.compile(sentence_model)
_, _, llm = setup_llm_pipeline()
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
    sessionId: str = "default"


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")


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

    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 6})
    return db, retriever


def get_conversation_memory(session_id: str):
    """세션 ID 기반으로 대화 메모리를 반환 (없으면 생성)"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = ConversationBufferWindowMemory(k=3, return_messages=True)
    return conversation_memory[session_id]


@app.get("/")
def home(api_key: str = Depends(verify_api_key)):
    return {"message": "Hello, World!"}


@app.post("/ask")
async def ask(request: QuestionRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()  # 시작 시간 기록

    question = request.question
    session_id = request.sessionId

    # 세션 ID 기반 대화 메모리 가져오기
    memory = get_conversation_memory(session_id)
    past_messages = memory.load_memory_variables({})["history"]
    history_text = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Model: {msg.content}"
        for msg in past_messages
    ])


    # 질문에서 카테고리 판별
    category = determine_category(question)
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    print(f"~category: {processing_time}")  

    if category not in FAISS_DB_PATHS:
        raise HTTPException(status_code=400, detail="No matching category found")

    # 해당 카테고리의 FAISS DB 로드
    db_path = FAISS_DB_PATHS[category]
    db, retriever = load_faiss_db(db_path)

    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    print(f"~faiss: {processing_time}")  

    # rag_chain = rag_llama(llm)
    rag_chain = rag_gemma(llm)


    # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):  # ✅ Flash Attention 활성화
    with torch.no_grad():  # 🔥 Inference에서 Gradient 계산 방지 (속도 향상)
        response = await asyncio.to_thread(rag_chain.invoke, {
            "history_text": history_text,  
            "context": retrieved_text,     
            "question": question
        })

    # 답변을 대화 메모리에 저장 (맥락 유지)
    memory.save_context({"input": question}, {"output": response.strip()})

    # 처리 시간 계산
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)  
    print(f"processing_time: {processing_time}")

    return {
        "category": category,
        "answer": response,
        "title": "Not yet bitch"
    }


# FastAPI 실행 (Uvicorn)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
