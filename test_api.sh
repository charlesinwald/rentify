#!/bin/bash

# Base URL for the API
BASE_URL="http://localhost:8000"
WS_URL="ws://localhost:8000"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
    fi
}

# Test user credentials
TEST_EMAIL="test@example.com"
TEST_PASSWORD="testpassword123"

# Test authentication endpoints
echo "Testing authentication endpoints..."

# Register a new user
echo "Testing user registration..."
REGISTER_RESPONSE=$(curl -s -X POST "${BASE_URL}/users/" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"${TEST_EMAIL}\",\"password\":\"${TEST_PASSWORD}\"}")
print_result $? "User registration"

# Login and get JWT token
echo "Testing user login..."
LOGIN_RESPONSE=$(curl -s -X POST "${BASE_URL}/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=${TEST_EMAIL}&password=${TEST_PASSWORD}")
TOKEN=$(echo $LOGIN_RESPONSE | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
print_result $? "User login"

if [ -z "$TOKEN" ]; then
    echo -e "${RED}Failed to get JWT token. Cannot proceed with authenticated tests.${NC}"
    exit 1
fi

# Test WebSocket chat
echo "Testing WebSocket chat..."
# Create a temporary Python script for WebSocket testing
cat > test_websocket.py << 'EOF'
import asyncio
import websockets
import json
import sys

async def test_chat():
    uri = f"ws://localhost:8000/ws/chat?token={sys.argv[1]}"
    async with websockets.connect(uri) as websocket:
        # Send a test message
        message = {"message": "Hello, this is a test message!"}
        await websocket.send(json.dumps(message))
        
        # Wait for the message to be broadcast back
        response = await websocket.recv()
        print(f"Received: {response}")
        
        # Send another message
        message = {"message": "This is another test message!"}
        await websocket.send(json.dumps(message))
        
        # Wait for the message to be broadcast back
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(test_chat())
EOF

# Run the WebSocket test
python3 test_websocket.py "$TOKEN"
print_result $? "WebSocket chat"

# Clean up the temporary script
rm test_websocket.py

# Test natural language search with authentication
echo "Testing natural language search with authentication..."
curl -s -X POST "${BASE_URL}/search/natural" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"query": "2 bedroom apartment in Manhattan", "page": 1, "limit": 10}' > /dev/null
print_result $? "Natural language search (authenticated)"

# Test complex natural language search
echo "Testing complex natural language search..."
curl -s -X POST "${BASE_URL}/search/natural" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"query": "3 bedroom apartment in Brooklyn with doorman and parking under 5000", "page": 1, "limit": 10}' > /dev/null
print_result $? "Complex natural language search"

# Test pagination
echo "Testing pagination..."
curl -s -X POST "${BASE_URL}/search/natural" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"query": "apartment", "page": 2, "limit": 10}' > /dev/null
print_result $? "Pagination"

# Test favorites functionality
echo "Testing favorites functionality..."

# Add a rental to favorites
echo "Adding rental to favorites..."
curl -s -X POST "${BASE_URL}/favorites/" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"rental_id": "1"}' > /dev/null
print_result $? "Add to favorites"

# Get user's favorites
echo "Getting user's favorites..."
curl -s -X GET "${BASE_URL}/favorites/" \
    -H "Authorization: Bearer ${TOKEN}" > /dev/null
print_result $? "Get favorites"

# Remove rental from favorites
echo "Removing rental from favorites..."
curl -s -X DELETE "${BASE_URL}/favorites/1" \
    -H "Authorization: Bearer ${TOKEN}" > /dev/null
print_result $? "Remove from favorites"

# Test search with filters
echo "Testing search with filters..."
curl -s -X GET "${BASE_URL}/search?borough=Manhattan&min_bedrooms=2&max_price=5000" \
    -H "Authorization: Bearer ${TOKEN}" > /dev/null
print_result $? "Search with filters"

# Test search with filters and verify photo URLs
echo "Testing search with filters and photo URLs..."
SEARCH_RESPONSE=$(curl -s -X GET "${BASE_URL}/search?borough=Manhattan&min_bedrooms=2&max_price=5000" \
    -H "Authorization: Bearer ${TOKEN}")

# Extract the first rental's photo URL
PHOTO_URL=$(echo $SEARCH_RESPONSE | jq -r '.results[0].photo_url')

if [ -n "$PHOTO_URL" ] && [ "$PHOTO_URL" != "null" ]; then
    echo "Testing photo URL: $PHOTO_URL"
    # Try to access the photo
    PHOTO_RESPONSE=$(curl -s -I "${BASE_URL}${PHOTO_URL}")
    if echo "$PHOTO_RESPONSE" | grep -q "200 OK"; then
        print_result 0 "Photo URL access"
        echo -e "${GREEN}Successfully accessed photo${NC}"
    else
        print_result 1 "Photo URL access"
        echo -e "${RED}Failed to access photo${NC}"
    fi
else
    print_result 1 "Photo URL presence"
    echo -e "${RED}No photo URL found in search results${NC}"
fi

# Test neighborhood endpoints
echo "Testing neighborhood endpoints..."

# Get all neighborhoods
echo "Getting all neighborhoods..."
curl -s -X GET "${BASE_URL}/neighborhoods" \
    -H "Authorization: Bearer ${TOKEN}" > /dev/null
print_result $? "Get all neighborhoods"

# Get neighborhoods by borough
echo "Getting neighborhoods by borough..."
curl -s -X GET "${BASE_URL}/neighborhoods/Manhattan" \
    -H "Authorization: Bearer ${TOKEN}" > /dev/null
print_result $? "Get neighborhoods by borough"

# Test image generation endpoint
echo "Testing image generation endpoint..."

# First, get a valid rental ID from search results
SEARCH_RESPONSE=$(curl -s -X GET "${BASE_URL}/search?borough=Manhattan&min_bedrooms=2&max_price=5000" \
    -H "Authorization: Bearer ${TOKEN}")

# Extract the first rental ID from the search results
RENTAL_ID=$(echo $SEARCH_RESPONSE | jq -r '.results[0].rental_id')

if [ -z "$RENTAL_ID" ] || [ "$RENTAL_ID" = "null" ]; then
    echo -e "${RED}No rental ID found in search results. Skipping image generation test...${NC}"
else
    echo "Using rental ID: $RENTAL_ID"
    # Get the image and save it directly
    curl -s -X GET "${BASE_URL}/rentals/${RENTAL_ID}/image" \
        -H "Authorization: Bearer ${TOKEN}" \
        --output "rental_${RENTAL_ID}.png"
    
    # Check if the file was created and has content
    if [ -s "rental_${RENTAL_ID}.png" ]; then
        print_result 0 "Image generation"
        echo -e "${GREEN}Image saved as rental_${RENTAL_ID}.png${NC}"
    else
        print_result 1 "Image generation"
        echo -e "${RED}Failed to save image${NC}"
    fi
fi

echo -e "\n${GREEN}All tests completed!${NC}" 