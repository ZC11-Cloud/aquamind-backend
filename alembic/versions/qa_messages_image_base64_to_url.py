"""Replace image_base64 with image_url for local file storage (OSS-ready)

Revision ID: qa_img_url
Revises: add_qa_image_b64
Create Date: 2026-03-10

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'qa_img_url'
down_revision: Union[str, Sequence[str], None] = 'add_qa_image_b64'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'qa_messages',
        sa.Column('image_url', sa.String(512), nullable=True),
    )
    op.drop_column('qa_messages', 'image_base64')


def downgrade() -> None:
    op.add_column(
        'qa_messages',
        sa.Column('image_base64', sa.Text(), nullable=True),
    )
    op.drop_column('qa_messages', 'image_url')
