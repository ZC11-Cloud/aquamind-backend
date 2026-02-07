import os

from dotenv import load_dotenv

load_dotenv()

DB_URL = "mysql+aiomysql://root:123456@localhost:3306/aquamind?charset=utf8mb4"

# JWT配置
SECRET_KEY = "cb2a0bb6fbe86bde71b847bde46937daeb58fd1af43fa54c730b12a5a86af318"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY环境变量未设置！")
