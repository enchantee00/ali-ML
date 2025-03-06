from fastapi import APIRouter, Depends

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dependencies import get_dependencies
from schemas import QuestionRequest
from services import *
from utils import verify_api_key

router = APIRouter()

@router.get("/")
def home(api_key: str = Depends(verify_api_key)):
    return {"message": "Hello, World!"}

@router.post("/ask")
async def ask(request: QuestionRequest, dependencies: dict = Depends(get_dependencies), api_key: str = Depends(verify_api_key)):
    return ask_llm(
        request,
        dependencies["model_manager"],
    )
    