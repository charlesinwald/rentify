#!/bin/bash

# Base URL
BASE_URL="http://localhost:8000"

echo "Testing Rentify API..."
echo "======================"

# Test root endpoint
echo -e "\n1. Testing root endpoint:"
curl -s "${BASE_URL}/" | jq '.'

# Test getting boroughs
echo -e "\n2. Testing get boroughs:"
curl -s "${BASE_URL}/boroughs" | jq '.'

# Test getting neighborhoods
echo -e "\n3. Testing get neighborhoods:"
curl -s "${BASE_URL}/neighborhoods" | jq '.'

# Test basic search with price range
echo -e "\n4. Testing search with price range ($2000-$4000):"
curl -s "${BASE_URL}/search?min_rent=2000&max_rent=4000" | jq '.'

# Test search with multiple filters
echo -e "\n5. Testing search with multiple filters (Manhattan, 1-2 bedrooms, $2000-$4000):"
curl -s "${BASE_URL}/search?borough=Manhattan&min_bedrooms=1&max_bedrooms=2&min_rent=2000&max_rent=4000" | jq '.'

# Test search with amenities
echo -e "\n6. Testing search with amenities (doorman and gym):"
curl -s "${BASE_URL}/search?has_doorman=true&has_gym=true" | jq '.'

# Test search with size and location
echo -e "\n7. Testing search with size and location (Upper East Side, min 800 sqft):"
curl -s "${BASE_URL}/search?neighborhood=Upper%20East%20Side&min_size_sqft=800" | jq '.'

# Test search with subway proximity
echo -e "\n8. Testing search with subway proximity (max 5 minutes):"
curl -s "${BASE_URL}/search?max_to_subway=5" | jq '.'

# Test complex search
echo -e "\n9. Testing complex search (Brooklyn, 2+ bedrooms, washer/dryer, no fee):"
curl -s "${BASE_URL}/search?borough=Brooklyn&min_bedrooms=2&has_washer_dryer=true&no_fee=true" | jq '.'

# Test search with all amenities
echo -e "\n10. Testing search with all amenities:"
curl -s "${BASE_URL}/search?has_doorman=true&has_elevator=true&has_dishwasher=true&has_washer_dryer=true&has_gym=true&has_roofdeck=true&has_patio=true" | jq '.'

# Test pagination
echo -e "\n11. Testing pagination (page 1, 5 items per page):"
curl -s "${BASE_URL}/search?page=1&page_size=5" | jq '.'

echo -e "\n12. Testing pagination (page 2, 5 items per page):"
curl -s "${BASE_URL}/search?page=2&page_size=5" | jq '.'

echo -e "\n13. Testing pagination with filters (Manhattan, page 1, 3 items per page):"
curl -s "${BASE_URL}/search?borough=Manhattan&page=1&page_size=3" | jq '.'

echo -e "\n14. Testing pagination with invalid page (should return last page):"
curl -s "${BASE_URL}/search?page=999&page_size=10" | jq '.'

# Test natural language search
echo -e "\n15. Testing natural language search (basic query):"
curl -s -X POST "${BASE_URL}/search/natural" \
  -H "Content-Type: application/json" \
  -d '{"query": "Find apartments in Manhattan under $3000 with at least 1 bedroom"}' | jq '.'

echo -e "\n16. Testing natural language search (complex query):"
curl -s -X POST "${BASE_URL}/search/natural" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me 2 bedroom apartments in Brooklyn with a doorman, gym, and washer/dryer, max 10 minutes to subway"}' | jq '.'

echo -e "\n17. Testing natural language search with pagination:"
curl -s -X POST "${BASE_URL}/search/natural" \
  -H "Content-Type: application/json" \
  -d '{"query": "Apartments in Queens with a patio", "page": 1, "page_size": 5}' | jq '.' 