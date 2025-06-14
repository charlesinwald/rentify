from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Optional, List, Dict
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_KEY'))

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
        model = genai.GenerativeModel('gemini-2.0-flash')
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

@app.post("/search/natural")
async def natural_language_search(search: NaturalLanguageSearch):
    # Parse the natural language query into structured filters
    filters = parse_natural_language_query(search.query)
    
    # Start with all data
    filtered_df = df.copy()
    
    # Apply filters
    if filters.get('min_rent') is not None:
        filtered_df = filtered_df[filtered_df['rent'] >= filters['min_rent']]
    if filters.get('max_rent') is not None:
        filtered_df = filtered_df[filtered_df['rent'] <= filters['max_rent']]
    if filters.get('min_bedrooms') is not None:
        filtered_df = filtered_df[filtered_df['bedrooms'] >= filters['min_bedrooms']]
    if filters.get('max_bedrooms') is not None:
        filtered_df = filtered_df[filtered_df['bedrooms'] <= filters['max_bedrooms']]
    if filters.get('min_bathrooms') is not None:
        filtered_df = filtered_df[filtered_df['bathrooms'] >= filters['min_bathrooms']]
    if filters.get('max_bathrooms') is not None:
        filtered_df = filtered_df[filtered_df['bathrooms'] <= filters['max_bathrooms']]
    if filters.get('min_size_sqft') is not None:
        filtered_df = filtered_df[filtered_df['size_sqft'] >= filters['min_size_sqft']]
    if filters.get('max_size_sqft') is not None:
        filtered_df = filtered_df[filtered_df['size_sqft'] <= filters['max_size_sqft']]
    if filters.get('min_to_subway') is not None:
        filtered_df = filtered_df[filtered_df['min_to_subway'] >= filters['min_to_subway']]
    if filters.get('max_to_subway') is not None:
        filtered_df = filtered_df[filtered_df['min_to_subway'] <= filters['max_to_subway']]
    if filters.get('borough'):
        filtered_df = filtered_df[filtered_df['borough'] == filters['borough']]
    if filters.get('neighborhood'):
        filtered_df = filtered_df[filtered_df['neighborhood'] == filters['neighborhood']]
    if filters.get('has_doorman') is not None:
        filtered_df = filtered_df[filtered_df['has_doorman'] == filters['has_doorman']]
    if filters.get('has_elevator') is not None:
        filtered_df = filtered_df[filtered_df['has_elevator'] == filters['has_elevator']]
    if filters.get('has_dishwasher') is not None:
        filtered_df = filtered_df[filtered_df['has_dishwasher'] == filters['has_dishwasher']]
    if filters.get('has_washer_dryer') is not None:
        filtered_df = filtered_df[filtered_df['has_washer_dryer'] == filters['has_washer_dryer']]
    if filters.get('has_gym') is not None:
        filtered_df = filtered_df[filtered_df['has_gym'] == filters['has_gym']]
    if filters.get('has_roofdeck') is not None:
        filtered_df = filtered_df[filtered_df['has_roofdeck'] == filters['has_roofdeck']]
    if filters.get('has_patio') is not None:
        filtered_df = filtered_df[filtered_df['has_patio'] == filters['has_patio']]
    if filters.get('no_fee') is not None:
        filtered_df = filtered_df[filtered_df['no_fee'] == filters['no_fee']]

    # Calculate pagination
    total_items = len(filtered_df)
    total_pages = (total_items + search.page_size - 1) // search.page_size
    
    # Adjust page if it's out of range
    if search.page > total_pages:
        search.page = total_pages if total_pages > 0 else 1
    
    # Get the slice for the current page
    start_idx = (search.page - 1) * search.page_size
    end_idx = start_idx + search.page_size
    page_df = filtered_df.iloc[start_idx:end_idx]

    # Convert to list of dictionaries
    results = page_df.to_dict(orient='records')
    
    return {
        "query": search.query,
        "parsed_filters": filters,
        "total_items": total_items,
        "total_pages": total_pages,
        "current_page": search.page,
        "page_size": search.page_size,
        "has_next": search.page < total_pages,
        "has_previous": search.page > 1,
        "results": results
    }

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
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page")
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
    
    return {
        "total_items": total_items,
        "total_pages": total_pages,
        "current_page": page,
        "page_size": page_size,
        "has_next": page < total_pages,
        "has_previous": page > 1,
        "results": results
    }

@app.get("/neighborhoods")
async def get_neighborhoods():
    return {"neighborhoods": sorted(df['neighborhood'].unique().tolist())}

@app.get("/boroughs")
async def get_boroughs():
    return {"boroughs": sorted(df['borough'].unique().tolist())} 