import logging
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from starlette.staticfiles import StaticFiles

from src.routers.user import router as user_router
from src.routers.qa import router as qa_router
from src.routers.image import router as image_router
from src.routers.knowledge import router as knowledge_router
from src.settings import DASHSCOPE_API_KEY, UPLOAD_DIR, KNOWLEDGE_UPLOAD_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
app = FastAPI()
CORS_ORIGINS=os.getenv("CORS_ORIGINS")
if not CORS_ORIGINS:
    raise ValueError("CORS_ORIGINS环境变量未设置！")
origins = CORS_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件访问（用于本地存储的头像），路径统一从 settings.UPLOAD_DIR 读取，避免在源路径再生成 static
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.include_router(user_router)
app.include_router(qa_router)
app.include_router(image_router)
app.include_router(knowledge_router)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)