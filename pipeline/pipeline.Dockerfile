FROM python:3.12-slim-bullseye

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get -y install wget p7zip-full

COPY ./requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

WORKDIR /pipeline
COPY ./pipeline /pipeline

CMD ["run-parts", "--verbose", "."]