"""Add reasoning_content column to qa_messages

Revision ID: add_qa_reasoning
Revises: add_qa_citations
Create Date: 2026-04-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'add_qa_reasoning'
down_revision: Union[str, Sequence[str], None] = 'add_qa_citations'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'qa_messages',
        sa.Column('reasoning_content', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('qa_messages', 'reasoning_content')
