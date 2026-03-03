# AquaMind Backend

水生生物智能助手后端服务：FastAPI + 通义千问 + YOLOv8 图像识别 + 知识库 RAG + Agent 对话。

## 技术栈

- **FastAPI**：Web 框架
- **通义千问 (DashScope)**：LLM 对话
- **YOLOv8 (Ultralytics)**：图像目标检测
- **LangChain / LangGraph**：Agent 编排与工具调用
- **ChromaDB**：知识库向量存储
- **SQLAlchemy + MySQL**：用户、会话等结构化数据

## 环境要求

- Python 3.x
- MySQL
- 通义千问 API Key
- （可选）YOLOv8 权重文件 `weights/best.pt`

## 环境变量

在项目根目录或 `backend` 下创建 `.env`，例如：

DB_URL=mysql+aiomysql://user:password@host:port/dbname
SECRET_KEY=你的JWT密钥
DASHSCOPE_API_KEY=你的通义千问APIKey
CORS_ORIGINS=http://localhost:3000
UPLOAD_DIR=backend/uploads
YOLO_WEIGHTS_PATH=weights/best.pt
KNOWLEDGE_UPLOAD_DIR=backend/uploads/knowledge
CHROMA_PERSIST_DIR=backend/data/chroma_kb
