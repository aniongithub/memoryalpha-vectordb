FROM python:3.12-slim-bullseye AS devcontainer

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get -y install curl wget git

COPY ./requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp