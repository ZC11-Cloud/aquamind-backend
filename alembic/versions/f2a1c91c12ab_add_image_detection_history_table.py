"""add image_detection_history table

Revision ID: f2a1c91c12ab
Revises: add_qa_reasoning
Create Date: 2026-05-01 18:58:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f2a1c91c12ab"
down_revision: Union[str, Sequence[str], None] = "add_qa_reasoning"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "image_detection_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("original_image_url", sa.String(length=512), nullable=True),
        sa.Column("annotated_image_url", sa.String(length=512), nullable=True),
        sa.Column("detections_json", sa.JSON(), nullable=False),
        sa.Column("top_species_name", sa.String(length=255), nullable=True),
        sa.Column("top_confidence", sa.Float(), nullable=True),
        sa.Column(
            "create_time",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "update_time",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_image_detection_history")),
    )
    op.create_index(
        op.f("ix_image_detection_history_user_id"),
        "image_detection_history",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_image_detection_history_user_id"),
        table_name="image_detection_history",
    )
    op.drop_table("image_detection_history")
