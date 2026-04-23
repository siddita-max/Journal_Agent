"""add activity description

Revision ID: 3bf19ed858b0
Revises: a21b097dea34
Create Date: 2026-04-23 08:20:21.671803

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3bf19ed858b0'
down_revision: Union[str, None] = 'a21b097dea34'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('image_records', sa.Column('activity_description', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('image_records', 'activity_description')
