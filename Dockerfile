# Use official Python image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run your Streamlit app on startup
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]