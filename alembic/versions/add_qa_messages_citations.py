"""Add citations column to qa_messages for knowledge base source references

Revision ID: add_qa_citations
Revises: qa_img_url
Create Date: 2026-03-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'add_qa_citations'
down_revision: Union[str, Sequence[str], None] = 'qa_img_url'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'qa_messages',
        sa.Column('citations', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('qa_messages', 'citations')
