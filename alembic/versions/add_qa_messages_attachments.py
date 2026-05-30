"""Add attachments column to qa_messages

Revision ID: add_qa_attachments
Revises: add_qa_reasoning
Create Date: 2026-05-29

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "add_qa_attachments"
down_revision: Union[str, Sequence[str], None] = "add_qa_reasoning"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("qa_messages", sa.Column("attachments", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("qa_messages", "attachments")
