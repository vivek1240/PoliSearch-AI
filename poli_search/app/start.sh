#!/bin/bash

# Wait until Elasticsearch is ready
echo "Waiting for Elasticsearch to be available..."
until curl -s http://elasticsearch:9200 >/dev/null; do
  sleep 5
done
echo "Elasticsearch is up and running!"

# Run the ingestion script to create the index and ingest documents
echo "Ingesting documents..."
python main.py --ingest

# Start the FastAPI app using Uvicorn
echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
