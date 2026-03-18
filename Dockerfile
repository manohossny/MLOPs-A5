# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements first for efficient layer caching
# (this layer is only rebuilt when dependencies change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code
COPY . .

# Run the training script by default
CMD ["python", "gan_mnist.py"]
