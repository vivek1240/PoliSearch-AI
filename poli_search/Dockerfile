# Use Python 3.10-slim as the base image (for better compatibility with libraries)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install curl and other dependencies
RUN apt-get update && apt-get install -y curl

# Copy the requirements file
COPY ./app/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the current directory contents into the container at /app
COPY ./app /app

# Set execution permissions to the startup script inside the container
RUN chmod +x /app/start.sh

# Expose port 8000
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["/app/start.sh"]
