FROM python:3.11-slim

WORKDIR /app

# Copy dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV USE_REAL_M1=false

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "m5_architect.main:app", "--host", "0.0.0.0", "--port", "8000"]
