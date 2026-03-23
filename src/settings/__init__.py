import os

from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise ValueError("DB_URL环境变量未设置！")
# JWT配置
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY环境变量未设置！")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10080  # 7 days

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY环境变量未设置！")

# 上传文件根目录（用于头像等静态文件，默认 uploads，避免在项目下生成 static 目录）
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "backend/uploads")

# YOLOv8 权重文件路径（绝对路径或相对于项目根目录，例如 weights/best.pt）
YOLO_WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS_PATH", "weights/best.pt")

KNOWLEDGE_UPLOAD_DIR = os.getenv("KNOWLEDGE_UPLOAD_DIR", "backend/uploads/knowledge")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "backend/data/chroma_kb")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
