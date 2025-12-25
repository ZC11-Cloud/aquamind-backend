from typing import Annotated, Literal, Optional, Any

from pydantic import BaseModel, Field

class ResponseSchema(BaseModel):
    """通用响应模型"""
    result: Annotated[Literal["success", "failure"], Field(default="success", description="操作的结果！")]
    code: int = Field(default=200, description="HTTP状态码")
    message: str = Field(default="操作成功", description="响应消息")
    data: Optional[Any] = Field(default=None, description="响应数据")
    meta: Optional[dict] = Field(default=None, description="元数据")