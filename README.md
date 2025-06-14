# Rentify API

A FastAPI backend for searching rental listings across Manhattan, Brooklyn, and Queens.

## Features

- Search rentals with multiple filters
- Filter by price, bedrooms, bathrooms, size, amenities, and more
- Get list of available neighborhoods and boroughs
- Dockerized for easy deployment

## Getting Started

### Prerequisites

- Docker
- Docker Compose (optional)

### Running with Docker

1. Build the Docker image:

```bash
docker build -t rentify-api .
```

2. Run the container:

```bash
docker run -p 8000:8000 rentify-api
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Search Rentals

```
GET /search
```

Query Parameters:

- `min_rent`: Minimum rent
- `max_rent`: Maximum rent
- `min_bedrooms`: Minimum number of bedrooms
- `max_bedrooms`: Maximum number of bedrooms
- `min_bathrooms`: Minimum number of bathrooms
- `max_bathrooms`: Maximum number of bathrooms
- `min_size_sqft`: Minimum square footage
- `max_size_sqft`: Maximum square footage
- `min_to_subway`: Minimum minutes to subway
- `max_to_subway`: Maximum minutes to subway
- `borough`: Borough name
- `neighborhood`: Neighborhood name
- `has_doorman`: Boolean
- `has_elevator`: Boolean
- `has_dishwasher`: Boolean
- `has_washer_dryer`: Boolean
- `has_gym`: Boolean
- `has_roofdeck`: Boolean
- `has_patio`: Boolean
- `no_fee`: Boolean

#### Get Neighborhoods

```
GET /neighborhoods
```

Returns a list of all available neighborhoods.

#### Get Boroughs

```
GET /boroughs
```

Returns a list of all available boroughs.

### Example Usage

Search for apartments in Manhattan with 1-2 bedrooms, rent between $2000-$4000:

```
GET /search?borough=Manhattan&min_bedrooms=1&max_bedrooms=2&min_rent=2000&max_rent=4000
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
