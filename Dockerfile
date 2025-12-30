FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy frontend build (should be in static/ from CI)
# If not present, app runs without frontend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# EXPERIMENT FAILURE FILE
#
# Append this line to the Dockerfile to cause a deployment (Docker build) failure.
# This is used on the experiment/deploy-fail branch.

RUN echo "EXPERIMENT: Intentional Docker build failure" && exit 1
