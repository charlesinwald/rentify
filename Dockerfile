FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install wait-for-it script
ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p alembic/versions

# Run database migrations and start the application
CMD ["/bin/bash", "-c", "/wait-for-it.sh $POSTGRES_HOST:$POSTGRES_PORT -- alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"] 