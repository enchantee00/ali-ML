import time

from fastapi import HTTPException
from langchain.schema import HumanMessage

from prompt import *
from schemas import QuestionRequest
from config import FAISS_DB_PATHS


def ask_llm(request: QuestionRequest, model_manager):
    start_time = time.time()  # 시작 시간 기록

    question = request.question
    session_id = request.sessionId

    # 세션 ID 기반 대화 메모리 가져오기
    memory = model_manager.get_conversation_memory(session_id)
    past_messages = memory.load_memory_variables({})["history"]
    history_text = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Model: {msg.content}"
        for msg in past_messages
    ])

    # 질문에서 카테고리 판별
    category = model_manager.determine_category(question)
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    print(f"~category: {processing_time}")  

    if category not in FAISS_DB_PATHS:
        raise HTTPException(status_code=400, detail="No matching category found")

    # 해당 카테고리의 FAISS DB 로드
    db_path = FAISS_DB_PATHS[category]
    db, retriever = model_manager.load_faiss_db(db_path)
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    print(f"~faiss: {processing_time}")

    title_prompt = vllm_llama_title()
    # response_prompt = vllm_llama_response()
    response_prompt = vllm_llama_response_without_history()


    # title, answer = await asyncio.gather(
    #     generate_title(session_id, title_prompt, question, sampling_params),
    #     generate_response(response_prompt, history_text, retrieved_text, question, sampling_params)
    # )

    title = model_manager.generate_title(session_id, title_prompt, question), 
    # answer = generate_response(response_prompt, history_text, retrieved_text, question, sampling_params)
    answer = model_manager.generate_response_without_history(response_prompt, retrieved_text, question)


    if isinstance(title, tuple):
        title = title[0]

    memory.save_context({"input": question}, {"output": answer})

    # 처리 시간 계산
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)  
    print(f"processing_time: {processing_time}")

    return {
        "category": category,
        "title": title,
        "answer": answer
    }