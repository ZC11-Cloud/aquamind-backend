"""Merge QA attachments and admin seed heads

Revision ID: merge_qa_attach_admin
Revises: add_qa_attachments, c3d9e6ab41f2
Create Date: 2026-05-29

"""
from typing import Sequence, Union


revision: str = "merge_qa_attach_admin"
down_revision: Union[str, Sequence[str], None] = (
    "add_qa_attachments",
    "c3d9e6ab41f2",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
