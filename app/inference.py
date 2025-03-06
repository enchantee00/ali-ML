import torch
import faiss
import re
import pickle

from vllm import LLM, SamplingParams
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory

from config import CATEGORY_DESCRIPTIONS, LLM_MODEL_PATH, EMBEDDING_MODEL_PATH

class ModelManager:
    def __init__(self):
        self.llm = None
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
        self.sentence_model = None
        self.embeddings = None

        self.category_index = None
        self.category_names = list(CATEGORY_DESCRIPTIONS.keys())
        self.category_sentences = list(CATEGORY_DESCRIPTIONS.values())

        self.title_memory = {}
        self.conversation_memory = {}

        self._load()
        
    def _load(self):
        # vLLM 초기화
        self.llm = LLM(model=LLM_MODEL_PATH, tensor_parallel_size=2, gpu_memory_utilization=0.8, max_model_len=4096)

        # 나머지 GPU 관련 객체 초기화
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") # or "cuda:n"
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        from sentence_transformers import SentenceTransformer

        sentence_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        self.sentence_model = torch.compile(sentence_model)
        
        # FAISS DB 관련 변수 초기화 등
        category_embeddings = sentence_model.encode(self.category_sentences, normalize_embeddings=True)
        embedding_size = category_embeddings.shape[1]
        category_index = faiss.IndexFlatL2(embedding_size)
        category_index.add(category_embeddings)  # 카테고리 벡터 추가
        self.category_index = category_index


    def determine_category(self, question: str) -> str:
        query_embedding = self.sentence_model.encode([question], normalize_embeddings=True)
        D, I = self.category_index.search(query_embedding, k=1)  # 가장 가까운 카테고리 검색
        return self.category_names[I[0][0]]  # 가장 가까운 카테고리 반환


    def load_faiss_db(self, faiss_db_directory: str):
        with open(faiss_db_directory + "_index_to_docstore_id.pkl", "rb") as f:
            index_to_docstore_id = pickle.load(f)

        with open(faiss_db_directory + "_docstore.pkl", "rb") as f:
            docstore = pickle.load(f)

        index = faiss.read_index(faiss_db_directory + "_faiss_db.index")

        db = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 5})
        return db, retriever


    def get_conversation_memory(self, session_id):
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = ConversationBufferWindowMemory(k=1, return_messages=True)
        return self.conversation_memory[session_id]


    def generate_response(self, prompt, history_text, retrieved_text, question):
        prompt = prompt.format(
            history_text=history_text,
            context=retrieved_text,
            question=question
        )
        # outputs = await asyncio.to_thread(llm.generate, [prompt], sampling_params)
        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response

    def generate_response_without_history(self, prompt, retrieved_text, question):
        prompt = prompt.format(
            context=retrieved_text,
            question=question
        )
        # outputs = await asyncio.to_thread(llm.generate, [prompt], sampling_params)
        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response


    def generate_title(self, session_id, prompt, question):
        if session_id in self.title_memory:
            return self.title_memory[session_id]
        
        prompt = prompt.format(
            question=question
        )
        # outputs = await asyncio.to_thread(llm.generate, [prompt], sampling_params)
        outputs = self.llm.generate([prompt], self.sampling_params)
        title = outputs[0].outputs[0].text.strip()
        title = re.sub(r"[^\w\s]", "", title)
        self.title_memory[session_id] = title 
        return title