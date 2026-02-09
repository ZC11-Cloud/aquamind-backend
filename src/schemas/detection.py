"""图像检测接口的请求/响应模型"""
from typing import List

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """单条检测结果"""
    class_name: str = Field(..., description="类别名称")
    class_id: int = Field(..., description="类别 ID")
    confidence: float = Field(..., ge=0, le=1, description="置信度 0~1")
    bbox: List[float] = Field(..., description="边界框 [x1, y1, x2, y2]")


class DetectionResponse(BaseModel):
    """检测接口响应"""
    detections: List[DetectionItem] = Field(default_factory=list, description="检测结果列表")
    count: int = Field(..., description="检测到的目标数量")
