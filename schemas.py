from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Chat schemas
class ChatMessage(BaseModel):
    message: str
    sender_email: str
    timestamp: datetime = datetime.now()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ChatMessageCreate(BaseModel):
    message: str

# Rental schemas
class RentalBase(BaseModel):
    rental_id: str
    rent: float
    bedrooms: int
    bathrooms: float
    has_doorman: bool
    has_elevator: bool
    has_dishwasher: bool
    has_washer_dryer: bool
    has_patio: bool
    has_gym: bool
    has_pool: bool
    has_central_air: bool
    has_fireplace: bool
    has_garage: bool
    has_parking: bool
    min_to_subway: int
    neighborhood: str
    borough: str
    photo_url: Optional[str] = None

    class Config:
        from_attributes = True

class RentalCreate(RentalBase):
    pass

class Rental(RentalBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Favorite schemas
class FavoriteBase(BaseModel):
    rental_id: str

class FavoriteCreate(FavoriteBase):
    pass

class Favorite(FavoriteBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Search schemas
class RentalSearch(BaseModel):
    min_rent: Optional[float] = None
    max_rent: Optional[float] = None
    min_bedrooms: Optional[float] = None
    max_bedrooms: Optional[float] = None
    min_bathrooms: Optional[float] = None
    max_bathrooms: Optional[float] = None
    min_size_sqft: Optional[float] = None
    max_size_sqft: Optional[float] = None
    min_to_subway: Optional[float] = None
    max_to_subway: Optional[float] = None
    borough: Optional[str] = None
    neighborhood: Optional[str] = None
    has_doorman: Optional[bool] = None
    has_elevator: Optional[bool] = None
    has_dishwasher: Optional[bool] = None
    has_washer_dryer: Optional[bool] = None
    has_gym: Optional[bool] = None
    has_roofdeck: Optional[bool] = None
    has_patio: Optional[bool] = None
    no_fee: Optional[bool] = None

class NaturalLanguageSearch(BaseModel):
    query: str
    page: int = 1
    page_size: int = 10
    photo_url: Optional[str] = None

# Response schemas
class PaginatedResponse(BaseModel):
    results: List[Rental]
    total: int
    page: int
    limit: int 