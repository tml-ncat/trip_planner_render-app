FROM python:3.10-slim

# Java for r5py + libexpat for rasterio
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openjdk-21-jdk-headless \
      libexpat1 \
 && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# App code
COPY src ./src

# ⬇️ Your data in repo root
COPY durham_new.osm.pbf ./durham_new.osm.pbf
COPY gtfs.zip           ./gtfs.zip

CMD ["gunicorn", "--chdir", "src", "app:server", "--workers", "1", "--threads", "4", "--timeout", "120"]
