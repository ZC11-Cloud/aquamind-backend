import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from starlette.staticfiles import StaticFiles

from routers.user import router as user_router
from routers.conversation import router as chat_router
from settings import DASHSCOPE_API_KEY

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
app = FastAPI()
origins = [
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件访问（用于本地存储的头像）
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.include_router(user_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/chat")
async def chat():
    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = llm.invoke("你好")
    return {"message": response.content}
