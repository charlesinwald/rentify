from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    favorites = relationship("Favorite", back_populates="user")

class Rental(Base):
    __tablename__ = "rentals"

    id = Column(Integer, primary_key=True, index=True)
    rental_id = Column(String, unique=True, index=True)
    rent = Column(Float)
    bedrooms = Column(Float)
    bathrooms = Column(Float)
    size_sqft = Column(Float)
    min_to_subway = Column(Float)
    floor = Column(Float)
    building_age_yrs = Column(Float)
    no_fee = Column(Boolean)
    has_roofdeck = Column(Boolean)
    has_washer_dryer = Column(Boolean)
    has_doorman = Column(Boolean)
    has_elevator = Column(Boolean)
    has_dishwasher = Column(Boolean)
    has_patio = Column(Boolean)
    has_gym = Column(Boolean)
    neighborhood = Column(String)
    borough = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    favorites = relationship("Favorite", back_populates="rental")

class Favorite(Base):
    __tablename__ = "favorites"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    rental_id = Column(Integer, ForeignKey("rentals.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="favorites")
    rental = relationship("Rental", back_populates="favorites") 