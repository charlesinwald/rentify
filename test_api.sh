#!/bin/bash

# Base URL for the API
BASE_URL="http://localhost:8000"

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

echo -e "\n${GREEN}All tests completed!${NC}" 