from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

from inference import ModelManager


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@asynccontextmanager
async def lifespan(app: FastAPI):    
    model_manager = ModelManager()
    app.state.model_manager = model_manager
    
    yield  # 애플리케이션 실행

    del model_manager
    print("서버 종료: 모델 리소스 해제")


def get_dependencies(request: Request):
    return {
        "model_manager": request.app.state.model_manager
    }