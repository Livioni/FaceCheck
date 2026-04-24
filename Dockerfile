FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt -r smirk/requirements.txt

ARG PYTORCH3D_WHL=
RUN if [ -n "$PYTORCH3D_WHL" ]; then pip install "$PYTORCH3D_WHL"; fi

EXPOSE 8000

CMD ["uvicorn", "facecheck.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
