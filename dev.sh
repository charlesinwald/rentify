#!/bin/bash

# Start development environment using Docker Compose

echo "Building and starting services with Docker Compose..."
docker-compose up --build

echo "To stop the services, press CTRL+C and then run: docker-compose down"