FROM python:3.10-slim

# OS deps: Java for r5py/JPype (use OpenJDK 21 on Debian trixie)
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-21-jdk-headless \
 && rm -rf /var/lib/apt/lists/*

# Point JAVA_HOME to JDK 21
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# App code
COPY src ./src

CMD ["gunicorn", "--chdir", "src", "app:server", "--workers", "1", "--threads", "4", "--timeout", "120"]
