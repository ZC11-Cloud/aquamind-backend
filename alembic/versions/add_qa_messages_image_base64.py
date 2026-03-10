"""add image_base64 to qa_messages

Revision ID: add_qa_image_b64
Revises: ab741e8e6744
Create Date: 2026-03-10

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'add_qa_image_b64'
down_revision: Union[str, Sequence[str], None] = 'ab741e8e6744'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'qa_messages',
        sa.Column('image_base64', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('qa_messages', 'image_base64')
