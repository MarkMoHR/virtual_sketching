FROM tensorflow/tensorflow:1.15.0-gpu-py3
ARG INSTALLDIR="/app/virtual_sketching"

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev libcairo2 python3-dev libffi-dev time \
    && \
    rm -rf /var/lib/apt/lists/*

COPY . $INSTALLDIR

WORKDIR $INSTALLDIR

RUN pip install -U pip wheel \
    && pip install -r requirements.txt \
    && pip cache purge

STOPSIGNAL SIGINT
ENTRYPOINT test_vectorization.py
