# Rentify API

A FastAPI application for searching rental listings in New York City with natural language processing capabilities.

## Features

- Natural language search for apartments using Google's Gemini API
- PostgreSQL database with SQLAlchemy ORM
- User authentication with JWT tokens
- Favorites system for saving preferred listings
- Paginated search results
- Filter by various criteria (price, bedrooms, amenities, etc.)
- Neighborhood and borough-based search

## Prerequisites

- Python 3.8+
- PostgreSQL
- Google Gemini API key

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
GEMINI_KEY=your_gemini_api_key_here
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rentify
JWT_SECRET_KEY=your_jwt_secret_key_here
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create PostgreSQL database:

```bash
createdb rentify
```

4. Run database migrations:

```bash
alembic upgrade head
```

5. Start the development server:

```bash
./dev.sh
```

## API Endpoints

### Authentication

- `POST /token` - Get JWT token for authentication
- `POST /users/` - Create new user account

### Search

- `POST /search/natural` - Natural language search
- `GET /search` - Advanced search with filters
- `GET /neighborhoods/` - List all neighborhoods
- `GET /neighborhoods/{borough}` - List neighborhoods by borough

### Favorites

- `POST /favorites/` - Add rental to favorites
- `GET /favorites/` - Get user's favorite rentals
- `DELETE /favorites/{rental_id}` - Remove rental from favorites

## Testing

Run the test script to verify API functionality:

```bash
./test_api.sh
```

## Docker

Build and run with Docker:

```bash
docker build -t rentify .
docker run -p 8000:8000 rentify
```

## License

MIT
