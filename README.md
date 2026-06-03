# AquaMind Backend

AquaMind 后端服务基于 FastAPI 构建，提供用户认证、AI 对话、知识库 RAG、图像识别、识别历史和管理员用户管理等接口。

## 技术栈

- FastAPI：后端 Web 框架。
- SQLAlchemy + Alembic：数据库 ORM 与迁移管理。
- MySQL：结构化数据存储。
- DashScope / 通义千问：大语言模型对话能力。
- LangChain / LangGraph：Agent 编排、工具调用和 RAG 流程组织。
- ChromaDB：知识库向量存储。
- Ultralytics YOLO：水生生物图像识别。

## 软件安装说明

### 环境要求

- Python 3.11。
- MySQL 8.x。
- DashScope API Key。
- YOLO 权重文件，默认使用 `weights/best.pt`。

### 安装步骤

进入后端目录：

```powershell
cd D:\bysj\AquaMind\backend
```

创建虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

创建 MySQL 数据库：

```sql
CREATE DATABASE aquamind DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

创建 `.env` 文件：

```env
DB_URL=mysql+aiomysql://user:password@localhost:3306/aquamind
SECRET_KEY=replace-with-your-secret-key
DASHSCOPE_API_KEY=replace-with-your-dashscope-api-key
CORS_ORIGINS=http://localhost:8001
UPLOAD_DIR=uploads
YOLO_WEIGHTS_PATH=weights/best.pt
KNOWLEDGE_UPLOAD_DIR=uploads/knowledge
QA_ATTACHMENT_UPLOAD_DIR=uploads/qa_attachments
CHROMA_PERSIST_DIR=data/chroma_kb
```

执行数据库迁移：

```powershell
alembic upgrade head
```

迁移完成后会写入默认管理员用户：

- 用户名：`admin`
- 密码：`ChangeMe123!`

也可以在执行迁移前通过环境变量 `DEFAULT_ADMIN_USERNAME`、`DEFAULT_ADMIN_PASSWORD`、`DEFAULT_ADMIN_REAL_NAME` 和 `DEFAULT_ADMIN_EMAIL` 自定义默认管理员信息。

启动服务：

```powershell
python run.py
```

服务默认运行在：

- 后端接口：`http://localhost:8000`
- 接口文档：`http://localhost:8000/docs`

## 软件使用说明

后端服务启动后，前端和接口客户端可以调用以下功能：

- 用户模块：注册、登录、获取当前用户信息、修改用户信息、管理员用户管理。
- AI 对话模块：创建会话、发送文本或图片消息、查询历史会话、查看引用来源。
- 知识库模块：上传文档、文档切分入库、标签筛选、全文检索、详情查看、下载和删除。
- 图像识别模块：上传图片检测、返回标注图和识别结果、查询历史记录、删除历史记录。
- 模型管理模块：上传新的图像识别模型，支持 `.pt`、`.onnx` 和 `.engine`。

建议先访问 `http://localhost:8000/docs` 验证接口是否可用，再启动前端进行完整流程测试。

## TensorRT 本地 engine 部署

TensorRT `.engine` 文件不能直接跨机器复用。如果 engine 文件是在另一台 GPU 设备上生成的，应在部署机器上重新生成，并将 `YOLO_WEIGHTS_PATH` 指向本机生成的 engine 文件。

从 `.pt` 权重生成本地 engine：

```powershell
python scripts/export_tensorrt_engines.py --source pt
```

如果当前 Python 环境未安装 TensorRT，可执行：

```powershell
python -m pip install --extra-index-url https://pypi.nvidia.com/ tensorrt
```

也可以通过 TensorRT `trtexec` 从 `.onnx` 生成：

```powershell
python scripts/export_tensorrt_engines.py --source onnx
```

脚本会在各模型目录下生成 `best_local.engine`，例如：

```env
YOLO_WEIGHTS_PATH=weights/yolov8/weights/best_local.engine
```

模型上传接口支持 `.pt`、`.onnx` 和 `.engine` 文件。
