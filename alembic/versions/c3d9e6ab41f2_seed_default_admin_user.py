"""seed default admin user

Revision ID: c3d9e6ab41f2
Revises: f2a1c91c12ab
Create Date: 2026-05-01 19:30:00.000000

"""
from typing import Sequence, Union
import os

from alembic import op
from sqlalchemy import text
from passlib.context import CryptContext


# revision identifiers, used by Alembic.
revision: str = "c3d9e6ab41f2"
down_revision: Union[str, Sequence[str], None] = "f2a1c91c12ab"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _read_seed_config() -> tuple[str, str, str | None, str | None]:
    username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
    password = os.getenv("DEFAULT_ADMIN_PASSWORD", "ChangeMe123!")
    real_name = os.getenv("DEFAULT_ADMIN_REAL_NAME", "系统管理员")
    email = os.getenv("DEFAULT_ADMIN_EMAIL")
    return username, password, real_name, email


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    username, password, real_name, email = _read_seed_config()
    exists = bind.execute(
        text("SELECT id FROM users WHERE username = :username LIMIT 1"),
        {"username": username},
    ).fetchone()

    if exists:
        return

    hashed_password = pwd_context.hash(password)
    bind.execute(
        text(
            """
            INSERT INTO users (username, password, real_name, email, role, status)
            VALUES (:username, :password, :real_name, :email, :role, :status)
            """
        ),
        {
            "username": username,
            "password": hashed_password,
            "real_name": real_name,
            "email": email,
            "role": 1,
            "status": 1,
        },
    )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    username, _, _, _ = _read_seed_config()
    bind.execute(
        text("DELETE FROM users WHERE username = :username AND role = :role"),
        {"username": username, "role": 1},
    )
