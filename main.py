from fastapi import FastAPI, Query, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Set
from pydantic import BaseModel
import pandas as pd
from datetime import timedelta, datetime
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import asyncio
import jwt
import base64
from io import BytesIO
from PIL import Image

from database import get_db
from models import User, Rental, Favorite
from schemas import (
    UserCreate, User as UserSchema,
    Token, Rental as RentalSchema,
    RentalSearch, NaturalLanguageSearch,
    Favorite as FavoriteSchema, FavoriteCreate,
    PaginatedResponse, ChatMessage, ChatMessageCreate
)
from auth import (
    verify_password, get_password_hash,
    create_access_token, get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Load environment variables
load_dotenv()

# Configure Gemini API for both text and image generation
gemini_key = os.getenv('GEMINI_KEY')
if not gemini_key:
    raise ValueError("GEMINI_KEY environment variable is not set")
genai.configure(api_key=gemini_key)
text_model = genai.GenerativeModel('gemini-2.0-flash')
image_model = genai.GenerativeModel('gemini-pro-vision')

app = FastAPI(title="Rentify API", description="API for searching rental listings")

# Mount the photos directory for static file serving
app.mount("/static/photos", StaticFiles(directory="photos"), name="photos")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        # Store active connections with their user emails
        self.active_connections: Dict[str, WebSocket] = {}
        # Store message history
        self.message_history: List[ChatMessage] = []

    async def connect(self, websocket: WebSocket, user_email: str):
        await websocket.accept()
        self.active_connections[user_email] = websocket
        # Send message history to the new connection
        for message in self.message_history[-50:]:  # Send last 50 messages
            message_dict = message.model_dump()
            message_dict["timestamp"] = message_dict["timestamp"].isoformat()
            await websocket.send_json(message_dict)

    def disconnect(self, user_email: str):
        if user_email in self.active_connections:
            del self.active_connections[user_email]

    async def broadcast(self, message: ChatMessage):
        # Add message to history
        self.message_history.append(message)
        # Keep only last 100 messages
        if len(self.message_history) > 100:
            self.message_history = self.message_history[-100:]
        
        # Convert message to dict with datetime serialization
        message_dict = message.model_dump()
        message_dict["timestamp"] = message_dict["timestamp"].isoformat()
        
        # Broadcast to all connected clients
        for connection in self.active_connections.values():
            try:
                await connection.send_json(message_dict)
            except WebSocketDisconnect:
                continue

manager = ConnectionManager()

# WebSocket endpoint for chat
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Verify token and get user
        try:
            payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
            user_email = payload.get("sub")
            if not user_email:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        except:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Connect to WebSocket
        await manager.connect(websocket, user_email)

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Create chat message
                chat_message = ChatMessage(
                    message=message_data["message"],
                    sender_email=user_email,
                    timestamp=datetime.now()
                )
                
                # Convert message to dict with datetime serialization
                message_dict = chat_message.model_dump()
                message_dict["timestamp"] = message_dict["timestamp"].isoformat()
                
                # Broadcast message
                await manager.broadcast(chat_message)
        except WebSocketDisconnect:
            manager.disconnect(user_email)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

# Load the CSV files
def load_data():
    manhattan_df = pd.read_csv('manhattan_with_photos.csv')
    brooklyn_df = pd.read_csv('brooklyn_with_photos.csv')
    queens_df = pd.read_csv('queens_with_photos.csv')
    
    # Combine the dataframes
    combined_df = pd.concat([manhattan_df, brooklyn_df, queens_df], ignore_index=True)
    
    # Update photo URLs to use the correct format
    combined_df['photo_url'] = combined_df['photo_url'].apply(
        lambda x: f"/photos/{x.split('/')[-2]}/{x.split('/')[-1]}" if pd.notna(x) else None
    )
    
    return combined_df

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
    Example response format:
    {{
        "min_bedrooms": 2,
        "has_gym": true,
        "borough": "Brooklyn"
    }}
    """

    try:
        print(f"\n=== Gemini Query Parsing Debug ===")
        print(f"Input query: {query}")
        print(f"Using model: {text_model.model_name}")
        
        # Generate content with safety settings
        response = text_model.generate_content(
            prompt,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        )
        
        # Extract the text content from the response
        response_text = response.text.strip()
        print(f"Raw response: {response_text}")
        
        # Remove any markdown code block markers if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        print(f"Cleaned response: {response_text}")
        
        # Parse the JSON response
        filters = json.loads(response_text)
        print(f"Parsed filters: {json.dumps(filters, indent=2)}")
        
        # Validate the filters
        if not isinstance(filters, dict):
            raise ValueError("Response is not a valid JSON object")
            
        # Convert borough to lowercase for consistency
        if 'borough' in filters:
            filters['borough'] = filters['borough'].lower()
            
        return filters
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        print(f"Failed to parse response: {response_text}")
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing natural language query response: {str(e)}"
        )
    except Exception as e:
        print(f"Error in parse_natural_language_query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing natural language query: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Welcome to Rentify API"}

@app.post("/search/natural", response_model=PaginatedResponse)
async def natural_language_search(
    search: NaturalLanguageSearch,
    db: Session = Depends(get_db)
):
    print(f"\n=== Natural Language Search Debug ===")
    print(f"Original query: {search.query}")
    
    # Parse the natural language query into structured filters
    try:
        filters = parse_natural_language_query(search.query)
        print(f"Parsed filters: {json.dumps(filters, indent=2)}")
    except Exception as e:
        print(f"Error parsing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing natural language query: {str(e)}"
        )
    
    # Create a RentalSearch object from the parsed filters
    try:
        rental_search = RentalSearch(**filters)
        print(f"Created RentalSearch object: {rental_search.model_dump()}")
    except Exception as e:
        print(f"Error creating RentalSearch object: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating search parameters: {str(e)}"
        )
    
    # Use the existing search endpoint with the parsed filters
    try:
        result = await search_rentals(
            min_rent=rental_search.min_rent,
            max_rent=rental_search.max_rent,
            min_bedrooms=rental_search.min_bedrooms,
            max_bedrooms=rental_search.max_bedrooms,
            min_bathrooms=rental_search.min_bathrooms,
            max_bathrooms=rental_search.max_bathrooms,
            min_size_sqft=rental_search.min_size_sqft,
            max_size_sqft=rental_search.max_size_sqft,
            min_to_subway=rental_search.min_to_subway,
            max_to_subway=rental_search.max_to_subway,
            borough=rental_search.borough,
            neighborhood=rental_search.neighborhood,
            has_doorman=rental_search.has_doorman,
            has_elevator=rental_search.has_elevator,
            has_dishwasher=rental_search.has_dishwasher,
            has_washer_dryer=rental_search.has_washer_dryer,
            has_gym=rental_search.has_gym,
            has_roofdeck=rental_search.has_roofdeck,
            has_patio=rental_search.has_patio,
            no_fee=rental_search.no_fee,
            page=search.page,
            page_size=search.page_size,
            db=db
        )
        print(f"Search results: {len(result.results)} items found")
        return result
    except Exception as e:
        print(f"Error performing search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing search: {str(e)}"
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
    print("\n=== Search Debug ===")
    print(f"Search parameters:")
    print(f"  borough: {borough}")
    print(f"  min_bedrooms: {min_bedrooms}")
    print(f"  max_bedrooms: {max_bedrooms}")
    
    # Start with all data
    filtered_df = df.copy()
    print(f"Initial dataset size: {len(filtered_df)} rows")
    
    # Apply filters
    if min_rent is not None:
        filtered_df = filtered_df[filtered_df['rent'] >= min_rent]
    if max_rent is not None:
        filtered_df = filtered_df[filtered_df['rent'] <= max_rent]
    if min_bedrooms is not None:
        filtered_df = filtered_df[filtered_df['bedrooms'] >= min_bedrooms]
        print(f"After min_bedrooms filter: {len(filtered_df)} rows")
    if max_bedrooms is not None:
        filtered_df = filtered_df[filtered_df['bedrooms'] <= max_bedrooms]
        print(f"After max_bedrooms filter: {len(filtered_df)} rows")
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
        filtered_df = filtered_df[filtered_df['borough'].str.lower() == borough.lower()]
        print(f"After borough filter: {len(filtered_df)} rows")
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

    print(f"Final filtered dataset size: {len(filtered_df)} rows")
    
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

    # Transform the data to match our Pydantic models
    results = []
    for _, row in page_df.iterrows():
        rental_dict = {
            'rental_id': str(row['rental_id']),
            'rent': float(row['rent']),
            'bedrooms': int(row['bedrooms']),
            'bathrooms': float(row['bathrooms']),
            'has_doorman': bool(row['has_doorman']),
            'has_elevator': bool(row['has_elevator']),
            'has_dishwasher': bool(row['has_dishwasher']),
            'has_washer_dryer': bool(row['has_washer_dryer']),
            'has_patio': bool(row['has_patio']),
            'has_gym': bool(row['has_gym']),
            'has_pool': bool(row.get('has_pool', False)),
            'has_central_air': bool(row.get('has_central_air', False)),
            'has_fireplace': bool(row.get('has_fireplace', False)),
            'has_garage': bool(row.get('has_garage', False)),
            'has_parking': bool(row.get('has_parking', False)),
            'min_to_subway': int(row['min_to_subway']),
            'neighborhood': str(row['neighborhood']),
            'borough': str(row['borough']),
            'photo_url': str(row['photo_url']) if pd.notna(row['photo_url']) else None,
            'id': int(row['rental_id']),
            'created_at': pd.Timestamp.now()
        }
        results.append(rental_dict)
    
    return PaginatedResponse(
        results=results,
        total=total_items,
        page=page,
        limit=page_size
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

@app.get("/rentals/{rental_id}/image")
async def generate_rental_image(rental_id: str):
    try:
        # Find the rental in the dataframe
        rental = df[df['rental_id'].astype(str) == str(rental_id)]
        
        if rental.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"Rental not found with ID: {rental_id}"
            )
        
        # Get rental details
        rental_data = rental.iloc[0]
        
        # Create a simple prompt focused only on image generation
        prompt = f"""Create an architectural visualization of a contemporary living space with {rental_data['bedrooms']} bedrooms and {rental_data['bathrooms']} bathrooms.
        Style: Modern architectural visualization, clean lines, minimalist design.
        View: Interior perspective of the main living area.
        Lighting: Natural daylight through large windows.
        Materials: Contemporary finishes, neutral color palette.
        Return only a base64 encoded image, nothing else.
        """
        
        # Generate image using Gemini
        response = image_model.generate_content(prompt)
        
        try:
            # Parse the response as JSON
            response_json = json.loads(response.text)
            
            # Extract the base64 image data
            base64_image = response_json.get('image', '')
            if base64_image.startswith('data:image/png;base64,'):
                base64_image = base64_image.replace('data:image/png;base64,', '')
            
            # Return only the base64 image
            return Response(content=base64_image, media_type="image/png")
            
        except json.JSONDecodeError:
            # If the response is not JSON, try to use it directly as base64
            if response.text.startswith('data:image/png;base64,'):
                base64_image = response.text.replace('data:image/png;base64,', '')
                return Response(content=base64_image, media_type="image/png")
            else:
                raise HTTPException(status_code=500, detail="Failed to generate image")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

# Add a photo endpoint to serve individual photos
@app.get("/photos/{category}/{filename}")
async def get_photo(category: str, filename: str):
    photo_path = f"photos/{category}/{filename}"
    if not os.path.exists(photo_path):
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(photo_path)

@app.get("/rentals/{rental_id}", response_model=RentalSchema)
async def get_rental(rental_id: str, db: Session = Depends(get_db)):
    # Find the rental in the dataframe
    rental = df[df['rental_id'].astype(str) == str(rental_id)]
    
    if rental.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Rental not found with ID: {rental_id}"
        )
    
    # Get rental details
    rental_data = rental.iloc[0]
    
    # Transform the data to match our Pydantic model
    rental_dict = {
        'rental_id': str(rental_data['rental_id']),
        'rent': float(rental_data['rent']),
        'bedrooms': int(rental_data['bedrooms']),
        'bathrooms': float(rental_data['bathrooms']),
        'has_doorman': bool(rental_data['has_doorman']),
        'has_elevator': bool(rental_data['has_elevator']),
        'has_dishwasher': bool(rental_data['has_dishwasher']),
        'has_washer_dryer': bool(rental_data['has_washer_dryer']),
        'has_patio': bool(rental_data['has_patio']),
        'has_gym': bool(rental_data['has_gym']),
        'has_pool': bool(rental_data.get('has_pool', False)),
        'has_central_air': bool(rental_data.get('has_central_air', False)),
        'has_fireplace': bool(rental_data.get('has_fireplace', False)),
        'has_garage': bool(rental_data.get('has_garage', False)),
        'has_parking': bool(rental_data.get('has_parking', False)),
        'min_to_subway': int(rental_data['min_to_subway']),
        'neighborhood': str(rental_data['neighborhood']),
        'borough': str(rental_data['borough']),
        'photo_url': str(rental_data['photo_url']) if pd.notna(rental_data['photo_url']) else None,
        'id': int(rental_data['rental_id']),
        'created_at': pd.Timestamp.now()
    }
    
    return rental_dict 