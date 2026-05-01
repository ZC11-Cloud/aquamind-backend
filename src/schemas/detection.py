"""图像检测接口的请求/响应模型"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """单条检测结果"""
    class_name: str = Field(..., description="类别名称")
    class_id: int = Field(..., description="类别 ID")
    confidence: float = Field(..., ge=0, le=1, description="置信度 0~1")
    bbox: List[float] = Field(..., description="边界框 [x1, y1, x2, y2]")
    species_name_zh: Optional[str] = Field(None, description="中文物种名称（LLM 增强）")
    description: Optional[str] = Field(None, description="简短描述（LLM 增强）")


class DetectionResponse(BaseModel):
    """检测接口响应"""
    detections: List[DetectionItem] = Field(default_factory=list, description="检测结果列表")
    count: int = Field(..., description="检测到的目标数量")
    annotated_image_url: Optional[str] = Field(None, description="标注后的图片 URL")
    original_image_url: Optional[str] = Field(None, description="原始上传图片 URL")


class DetectionHistoryItem(BaseModel):
    """识别历史记录项。"""
    id: int = Field(..., description="历史记录 ID")
    user_id: int = Field(..., description="用户 ID")
    original_image_url: Optional[str] = Field(None, description="原图 URL")
    annotated_image_url: Optional[str] = Field(None, description="标注图 URL")
    detections: List[DetectionItem] = Field(default_factory=list, description="检测结果")
    top_species_name: Optional[str] = Field(None, description="主识别物种名称")
    top_confidence: Optional[float] = Field(None, description="主识别置信度")
    create_time: datetime = Field(..., description="创建时间")


class DetectionHistoryListResponse(BaseModel):
    """识别历史分页响应。"""
    records: List[DetectionHistoryItem] = Field(default_factory=list, description="记录列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="页码")
    page_size: int = Field(..., description="每页数量")


class CurrentModelResponse(BaseModel):
    """当前模型信息。"""
    weights_name: str = Field(..., description="权重文件名")
    weights_path: str = Field(..., description="权重绝对路径")
    updated_at: Optional[str] = Field(None, description="最近更新时间")
