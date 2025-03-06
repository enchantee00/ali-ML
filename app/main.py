import os
import uvicorn
from fastapi import FastAPI

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dependencies import lifespan
from routers import router

app = FastAPI(lifespan=lifespan)

# 라우터 등록
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)