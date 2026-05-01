from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, JSON, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from src.models import Base


class ImageDetectionHistory(Base):
    __tablename__ = "image_detection_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    original_image_url: Mapped[str | None] = mapped_column(String(512), default=None)
    annotated_image_url: Mapped[str | None] = mapped_column(String(512), default=None)
    detections_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    top_species_name: Mapped[str | None] = mapped_column(String(255), default=None)
    top_confidence: Mapped[float | None] = mapped_column(Float, default=None)
    create_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=func.now(),
    )
