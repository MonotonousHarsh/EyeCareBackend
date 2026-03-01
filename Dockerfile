# Use Python 3.11 to avoid the Pydantic crash we saw earlier!
FROM python:3.11-slim

# Hugging Face Spaces require the app to run as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /home/user/app

# Copy the requirements file first to save build time
COPY --chown=user:user requirements.txt .

# Install the heavy AI dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all your backend code and models
COPY --chown=user:user . .

# Hugging Face exposes port 7860 by default
EXPOSE 7860

# Start the FastAPI server on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]