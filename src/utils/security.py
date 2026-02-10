import logging

from jose import JWTError, jwt
from passlib.context import CryptContext
from src.settings import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import datetime, timedelta, timezone
from typing import Optional

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
logger = logging.getLogger(__name__)

# bcrypt 最大支持 72 字节，超过会报错
BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    pwd_bytes = len(password.encode("utf-8"))
    if pwd_bytes > BCRYPT_MAX_BYTES:
        logger.warning("hash_password: 密码长度 %d 字节，超过 bcrypt 限制 %d，将截断", pwd_bytes, BCRYPT_MAX_BYTES)
    try:
        return pwd_context.hash(password)
    except ValueError as e:
        logger.exception("hash_password 失败: %s", e)
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    pwd_bytes = len(plain_password.encode("utf-8"))
    if pwd_bytes > BCRYPT_MAX_BYTES:
        logger.warning("verify_password: 明文密码长度 %d 字节，超过 bcrypt 限制 %d", pwd_bytes, BCRYPT_MAX_BYTES)
    try:
        ok = pwd_context.verify(plain_password, hashed_password)
        logger.debug("verify_password: 结果=%s, 明文长度=%d 字节", ok, pwd_bytes)
        return ok
    except ValueError as e:
        logger.exception("verify_password 失败（如密码超过72字节）: %s", e)
        raise


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """解码访问令牌"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
