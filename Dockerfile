# Live dashboard on a server. NOT yet built or run (written on a Mac without Docker): treat it as a starting point.
# Build from the project folder so that models/ (the trained eye model is not in git) is copied in:
#   docker build -t drowsiness-dashboard .  &&  docker run -p 7860:7860 drowsiness-dashboard
# The page needs https to open the camera, so run it behind a host that provides https.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY download_models.py .
COPY models ./models
RUN python download_models.py
COPY drowsiness ./drowsiness
COPY apps ./apps
EXPOSE 7860
# the camera modes; recordings are not in the image (data/ is not copied)
CMD ["python", "apps/multimodal.py", "--host", "0.0.0.0", "--port", "7860"]
