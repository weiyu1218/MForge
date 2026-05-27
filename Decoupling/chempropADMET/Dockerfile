FROM mambaorg/micromamba:1.5-jammy

USER root
WORKDIR /app

COPY environment.yml /app/environment.yml
RUN micromamba install -y -n base -f environment.yml && micromamba clean -a -y

COPY . /app

# Create model directory
RUN mkdir -p /app/models

EXPOSE 8901

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8901"]
