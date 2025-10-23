FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN cmake -S . -B build -DUSE_XDRFILE=OFF -DUSE_TNG=OFF && \
    cmake --build build -- -j$(nproc)

CMD ["/app/build/bls_analyze", "--help"]

