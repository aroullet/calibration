# hash:sha256:bc4a77a4e57cae93a5ccd15570a3a3c2c42177b1749cd6d3e0e6999c89089096
FROM registry.codeocean.com/codeocean/miniconda3:4.3.30-python3.6.3-ubuntu16.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.1ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir --upgrade-strategy=only-if-needed \
    keras==2.2.4 \
    keras-vis==0.4.1 \
    scikit-learn==0.23.2 \
    tensorflow==1.11.0

COPY postInstall /
RUN /postInstall
