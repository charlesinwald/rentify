from fastapi import FastAPI, Query, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Optional, List, Dict
from pydantic import BaseModel
import pandas as pd
from datetime import timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

from database import get_db
from models import User, Rental, Favorite
from schemas import (
    UserCreate, User as UserSchema,
    Token, Rental as RentalSchema,
    RentalSearch, NaturalLanguageSearch,
    Favorite as FavoriteSchema, FavoriteCreate,
    PaginatedResponse
)
from auth import (
    verify_password, get_password_hash,
    create_access_token, get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI(title="Rentify API", description="API for searching rental listings")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the CSV files
def load_data():
    manhattan_df = pd.read_csv('manhattan.csv')
    brooklyn_df = pd.read_csv('brooklyn.csv')
    queens_df = pd.read_csv('queens.csv')
    return pd.concat([manhattan_df, brooklyn_df, queens_df], ignore_index=True)

# Load data at startup
df = load_data()

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

def parse_natural_language_query(query: str) -> Dict:
    """Use Gemini to parse natural language query into structured filters."""
    prompt = f"""
    Parse the following apartment search query into structured filters. Return a JSON object with the following optional fields:
    - min_rent: float
    - max_rent: float
    - min_bedrooms: float
    - max_bedrooms: float
    - min_bathrooms: float
    - max_bathrooms: float
    - min_size_sqft: float
    - max_size_sqft: float
    - min_to_subway: float
    - max_to_subway: float
    - borough: string
    - neighborhood: string
    - has_doorman: boolean
    - has_elevator: boolean
    - has_dishwasher: boolean
    - has_washer_dryer: boolean
    - has_gym: boolean
    - has_roofdeck: boolean
    - has_patio: boolean
    - no_fee: boolean

    Query: {query}

    Only include fields that are explicitly mentioned in the query. Return valid JSON only.
    """

    try:
        response = model.generate_content(prompt)
        # Extract the text content from the response
        response_text = response.text.strip()
        # Remove any markdown code block markers if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        filters = json.loads(response_text)
        return filters
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing natural language query: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to Rentify API"}

@app.post("/search/natural", response_model=PaginatedResponse)
async def natural_language_search(
    search: NaturalLanguageSearch,
    db: Session = Depends(get_db)
):
    # Parse the natural language query into structured filters
    filters = parse_natural_language_query(search.query)
    
    # Create a RentalSearch object from the parsed filters
    rental_search = RentalSearch(**filters)
    
    # Use the existing search endpoint with the parsed filters
    return search_rentals(
        search=rental_search,
        page=search.page,
        page_size=search.page_size,
        db=db
    )

@app.get("/search")
async def search_rentals(
    min_rent: Optional[float] = None,
    max_rent: Optional[float] = None,
    min_bedrooms: Optional[float] = None,
    max_bedrooms: Optional[float] = None,
    min_bathrooms: Optional[float] = None,
    max_bathrooms: Optional[float] = None,
    min_size_sqft: Optional[float] = None,
    max_size_sqft: Optional[float] = None,
    min_to_subway: Optional[float] = None,
    max_to_subway: Optional[float] = None,
    borough: Optional[str] = None,
    neighborhood: Optional[str] = None,
    has_doorman: Optional[bool] = None,
    has_elevator: Optional[bool] = None,
    has_dishwasher: Optional[bool] = None,
    has_washer_dryer: Optional[bool] = None,
    has_gym: Optional[bool] = None,
    has_roofdeck: Optional[bool] = None,
    has_patio: Optional[bool] = None,
    no_fee: Optional[bool] = None,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    db: Session = Depends(get_db)
):
    # Start with all data
    filtered_df = df.copy()
    
    # Apply filters
    if min_rent is not None:
        filtered_df = filtered_df[filtered_df['rent'] >= min_rent]
    if max_rent is not None:
        filtered_df = filtered_df[filtered_df['rent'] <= max_rent]
    if min_bedrooms is not None:
        filtered_df = filtered_df[filtered_df['bedrooms'] >= min_bedrooms]
    if max_bedrooms is not None:
        filtered_df = filtered_df[filtered_df['bedrooms'] <= max_bedrooms]
    if min_bathrooms is not None:
        filtered_df = filtered_df[filtered_df['bathrooms'] >= min_bathrooms]
    if max_bathrooms is not None:
        filtered_df = filtered_df[filtered_df['bathrooms'] <= max_bathrooms]
    if min_size_sqft is not None:
        filtered_df = filtered_df[filtered_df['size_sqft'] >= min_size_sqft]
    if max_size_sqft is not None:
        filtered_df = filtered_df[filtered_df['size_sqft'] <= max_size_sqft]
    if min_to_subway is not None:
        filtered_df = filtered_df[filtered_df['min_to_subway'] >= min_to_subway]
    if max_to_subway is not None:
        filtered_df = filtered_df[filtered_df['min_to_subway'] <= max_to_subway]
    if borough:
        filtered_df = filtered_df[filtered_df['borough'] == borough]
    if neighborhood:
        filtered_df = filtered_df[filtered_df['neighborhood'] == neighborhood]
    if has_doorman is not None:
        filtered_df = filtered_df[filtered_df['has_doorman'] == has_doorman]
    if has_elevator is not None:
        filtered_df = filtered_df[filtered_df['has_elevator'] == has_elevator]
    if has_dishwasher is not None:
        filtered_df = filtered_df[filtered_df['has_dishwasher'] == has_dishwasher]
    if has_washer_dryer is not None:
        filtered_df = filtered_df[filtered_df['has_washer_dryer'] == has_washer_dryer]
    if has_gym is not None:
        filtered_df = filtered_df[filtered_df['has_gym'] == has_gym]
    if has_roofdeck is not None:
        filtered_df = filtered_df[filtered_df['has_roofdeck'] == has_roofdeck]
    if has_patio is not None:
        filtered_df = filtered_df[filtered_df['has_patio'] == has_patio]
    if no_fee is not None:
        filtered_df = filtered_df[filtered_df['no_fee'] == no_fee]

    # Calculate pagination
    total_items = len(filtered_df)
    total_pages = (total_items + page_size - 1) // page_size
    
    # Adjust page if it's out of range
    if page > total_pages:
        page = total_pages if total_pages > 0 else 1
    
    # Get the slice for the current page
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = filtered_df.iloc[start_idx:end_idx]

    # Convert to list of dictionaries
    results = page_df.to_dict(orient='records')
    
    return PaginatedResponse(
        total_items=total_items,
        total_pages=total_pages,
        current_page=page,
        page_size=page_size,
        has_next=page < total_pages,
        has_previous=page > 1,
        results=results
    )

@app.get("/neighborhoods")
async def get_neighborhoods():
    return {"neighborhoods": sorted(df['neighborhood'].unique().tolist())}

@app.get("/boroughs")
async def get_boroughs():
    return {"boroughs": sorted(df['borough'].unique().tolist())}

# Authentication endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=UserSchema)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Favorites endpoints
@app.post("/favorites/", response_model=FavoriteSchema)
def create_favorite(
    favorite: FavoriteCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if rental exists
    rental = db.query(Rental).filter(Rental.id == favorite.rental_id).first()
    if not rental:
        raise HTTPException(status_code=404, detail="Rental not found")
    
    # Check if favorite already exists
    existing_favorite = db.query(Favorite).filter(
        Favorite.user_id == current_user.id,
        Favorite.rental_id == favorite.rental_id
    ).first()
    if existing_favorite:
        raise HTTPException(status_code=400, detail="Rental already in favorites")
    
    # Create new favorite
    db_favorite = Favorite(
        user_id=current_user.id,
        rental_id=favorite.rental_id
    )
    db.add(db_favorite)
    db.commit()
    db.refresh(db_favorite)
    return db_favorite

@app.get("/favorites/", response_model=List[RentalSchema])
def get_favorites(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    favorites = db.query(Favorite).filter(Favorite.user_id == current_user.id).all()
    rental_ids = [f.rental_id for f in favorites]
    rentals = db.query(Rental).filter(Rental.id.in_(rental_ids)).all()
    return rentals

@app.delete("/favorites/{rental_id}")
def delete_favorite(
    rental_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    favorite = db.query(Favorite).filter(
        Favorite.user_id == current_user.id,
        Favorite.rental_id == rental_id
    ).first()
    if not favorite:
        raise HTTPException(status_code=404, detail="Favorite not found")
    
    db.delete(favorite)
    db.commit()
    return {"message": "Favorite removed successfully"}

@app.get("/neighborhoods/")
def get_neighborhoods():
    return sorted(df['neighborhood'].unique().tolist())

@app.get("/neighborhoods/{borough}")
def get_neighborhoods_by_borough(borough: str):
    borough_data = df[df['borough'].str.lower() == borough.lower()]
    return sorted(borough_data['neighborhood'].unique().tolist()) 