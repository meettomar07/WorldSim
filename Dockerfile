FROM python:3.11-slim

WORKDIR /app

# Copy repository contents
COPY . /app

# Expose default port
ENV PORT=8000
EXPOSE 8000

# Start WorldSim Live 2D Viewer
CMD ["sh", "-c", "python run_viewer.py --host 0.0.0.0 --port ${PORT:-8000} --no-browser"]
