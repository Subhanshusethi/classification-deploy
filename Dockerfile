# 1. Use an official Python base image
FROM python:3.12.1-slim-bookworm

# 2. Set a working directory inside the container
WORKDIR /mldeploy

# 3. Install system packages required for image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 4. Copy your local files into the container
COPY . /mldeploy/

# 5. Install Python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# 6. Expose the port your FastAPI app will run on (default: 8000)
EXPOSE 8000

# 7. Command to run your FastAPI app using uvicorn
WORKDIR /mldeploy
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
