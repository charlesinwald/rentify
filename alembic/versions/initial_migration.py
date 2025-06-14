"""initial migration

Revision ID: 001
Revises: 
Create Date: 2024-03-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )

    # Create rentals table
    op.create_table(
        'rentals',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rental_id', sa.String(), nullable=False),
        sa.Column('rent', sa.Float(), nullable=False),
        sa.Column('bedrooms', sa.Float(), nullable=False),
        sa.Column('bathrooms', sa.Float(), nullable=False),
        sa.Column('size_sqft', sa.Float(), nullable=False),
        sa.Column('min_to_subway', sa.Float(), nullable=False),
        sa.Column('floor', sa.Float(), nullable=False),
        sa.Column('building_age_yrs', sa.Float(), nullable=False),
        sa.Column('no_fee', sa.Boolean(), nullable=False),
        sa.Column('has_roofdeck', sa.Boolean(), nullable=False),
        sa.Column('has_washer_dryer', sa.Boolean(), nullable=False),
        sa.Column('has_doorman', sa.Boolean(), nullable=False),
        sa.Column('has_elevator', sa.Boolean(), nullable=False),
        sa.Column('has_dishwasher', sa.Boolean(), nullable=False),
        sa.Column('has_patio', sa.Boolean(), nullable=False),
        sa.Column('has_gym', sa.Boolean(), nullable=False),
        sa.Column('neighborhood', sa.String(), nullable=False),
        sa.Column('borough', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('rental_id')
    )

    # Create favorites table
    op.create_table(
        'favorites',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('rental_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['rental_id'], ['rentals.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'rental_id', name='unique_user_rental')
    )


def downgrade() -> None:
    op.drop_table('favorites')
    op.drop_table('rentals')
    op.drop_table('users') 