from fastapi import FastAPI
from vllm import LLM, SamplingParams
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import torch
# from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.vectorstores import FAISS

torch.cuda.empty_cache()




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

API_KEY = "RIS-alitouch-key"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 워커 프로세스에서만 실행됨
    global llm
    # 여기서 GPU 관련 초기화
    llm = LLM(model="MarkrAI/RAG-KO-Mixtral-7Bx2-v2.1", tensor_parallel_size=2)

    model_path = "nlpai-lab/KURE-v1"
    model_kwargs = {"device": "cuda:3"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # sentence_model = SentenceTransformer(model_path)

    yield
    # 종료 시 정리 작업 수행
    llm = None
    print("서버 종료")

app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(request: PromptRequest):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

    template = """<bos><start_of_turn>system
    당신은 법률 전문가 AI입니다. 참고 문서를 바탕으로 정확하고 논리적인 답변을 제공하세요.
    - 질문과 관련된 법률 조항만 요약하여 답변하세요.
    - 답변은 명확하고 완결된 문장으로 구성하세요.
    - 동일한 정보를 반복하지 마세요.

    <end_of_turn>
    <start_of_turn>user

    참고 문서 내용:
    {context}

    질문:
    {question}

    명확하고 논리적인 문장으로 답변하세요. 질문에 대한 핵심 개념을 포함하여 일관된 문장으로 설명하세요.
    <end_of_turn>
    <start_of_turn>model
    """
    retrieved_text='사용자가 외국인근로자를 고용하려는 경우에는 표준근로계약서[「외국인근로자의 고용 등에 관한 법률 시행규칙」\n별지 제6호서식( 농업·축산업·어업분야는 「외국인근로자의 고용 등에 관한 법률 시행규칙」 별지 제6호의2서식)]를\n사용해서 근로계약을 체결해야 합니다(\n「외국인근로자의 고용 등에 관한 법률」 제9조제1항 및 제12조제1항).\n√ 사용자는 근로계약의 체결을 한국산업인력공단에 대행하게 할 수 있습니다(\n「외국인근로자의 고용 등에 관\n한 법률」 제9조제2항 및 제12조제1항).\n 사용자 또는 한국산업인력공단이 근로계약을 체결하거나 이를 대행하는 경우에는 근로계약서 2부를 작성하고 그\n중 1부를 외국인근로자에게 내주어야 합니다(「외국인근로자의 고용 등에 관한 법률 시행령」 제16조).\n 근로계약기간\n 외국인근로자와 사용자는 3년의 기간 내에서 당사자 간 합의에 따라 근로계약을 체결하거나 갱신할 수 있습니다(\n「외국인근로자의 고용 등에 관한 법률」 제9조제3항, 제12조제1항 및 제18조).\n 다만, 취업활동기간 3년이 만료되어 출국하기 전에 사용자가 고용노동부장관에게 재고용허가를 요청한 외국인근로\n자는 3년의 기간제한(「외국인근로자의 고용 등에 관한 법률」 제18조)에도 불구하고 한 차례만 2년 미만의 범위에서\n취업활동기간을 연장받아, 연장된 취업활동기간의 범위에서 근로계약을 체결할 수 있습니다(「외국인근로자의 고용\n등에 관한 법률」 제18조의2제1항).\n25. 1. 31. 오후 4:30\n에만 고용허가를 신청할 수 있습니다(\n「외국인근로자의 고용 등에 관한 법률」 제8조제1항).\n 다만, 사용자가 고용센터의 소개에도 불구하고 정당한 이유없이 2회 이상 채용을 거부 하였다면 내국인 구인노력을\n한 것으로 인정되지 않습니다(\n「외국인근로자의 고용 등에 관한 법률 시행령」 제13조의4제2호 단서).\n 고용허가의 신청기한과 제출서류\n· 사용자가 내국인 구인노력을 했음에도 불구하고 내국인을 채용하지 못하면 다음의 서류를 구인노력기간이 지난 후 3\n개월 이내에 사업 또는 사업장의 소재지를 관할하는 고용센터 소장에게 제출함으로써 외국인근로자의 고용허가를\n신청할 수 있습니다(「외국인근로자의 고용 등에 관한 법률 시행규칙」 제5조제1항, 별지 제4호서식, 별지 제4호의2\n서식 및 \n「외국인근로자의 고용 등에 관한 법률 시행령」 제13조의4제1호).\n1. 외국인근로자 고용허가서 발급신청서\n2. 정책위원회에서 정한 외국인근로자의 도입 업종, 외국인근로자를 고용할 수 있는 사업 또는 사업장에 해당함을 증\n명할 수 있는 서류\n3. 농어업인안전보험 가입 확약서(「산업재해보상보험법」 및 「어선원 및 어선 재해보상보험법」을 적용받지 않는 사업\n또는 사업장만 제출함)\n 외국인구직자의 추천\n국외에 있는 비전문취업(E-9) 체류자격 외국인근로자를 고용하려는 경우의 절차\xa0\xa0\n25. 1. 31. 오후 4:30\n외국인근로자 고용·취업 > 외국인근로자 고용 > 외국인근로자 고용절차 > 비전문취업(E-9) 체류자격자 고용절차 (본문) | 찾…\n행규칙」 제14조의2제2항).\n 고용노동부장관은 감염병 확산, 천재지변 등의 사유로 외국인근로자의 입국과 출국이 어렵다고 인정되는 경우에는\n외국인력정책위원회의 심의·의결을 거쳐 1년의 범위에서 취업활동 기간을 연장할 수 있습니다(「외국인근로자의 고\n용 등에 관한 법률」 제18조의2제2항).\n근로 시작\n 이상의 절차를 마치면 근로계약을 체결한 외국인근로자를 사업장에 배치시켜 근로를 시작하게 됩니다.\n근로개시 신고\n 사용자는 외국인근로자가 근로를 개시한 날부터 14일 이내에 특례고용 외국인근로자 근로개시 신고서(「외국인근로자\n의 고용 등에 관한 법률 시행규칙」 제11호 서식)에 그 사실을 기재하고 다음의 서류를 첨부하여 사용자가 영위하는 사\n업 또는 사업장의 소재지를 관할하는 고용센터 소장에게 제출해야 합니다(\n「외국인근로자의 고용 등에 관한 법률」\n제12조제4항 및 「외국인근로자의 고용 등에 관한 법률 시행규칙」 제12조의3).\n1. 표준근로계약서 사본\n2. 외국인등록증 사본 또는 여권 사본\n특례고용가능확인 변경확인\n 특례고용가능 변경확인 사유\n 사용자가 특례고용가능확인서를 발급받은 후 해당 사업 또는 사업장의 업종 또는 규모 등의 변화로 특례고용확인서\n의 내용 중 다음의 어느 하나에 해당하는 사항을 변경해야 할 필요가 있으면 고용센터 소장에게 특례고용가능확인\n서의 변경확인을 받아야 합니다(\n「외국인근로자의 고용 등에 관한 법률 시행령」 제20조의2 및 \n「외국인근로\n자의 고용 등에 관한 법률 시행규칙」 제13조제1항).'


    formatted_prompt = template.format(
        context=retrieved_text,
        question=request.prompt
    )

    outputs = llm.generate([formatted_prompt], sampling_params)
    return {"response": outputs[0].outputs[0].text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
