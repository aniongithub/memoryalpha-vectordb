FROM python:3.12-slim-bullseye as devcontainer

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get -y install wget p7zip-full git

COPY ./pipeline/pipeline-requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir -r /tmp/pip-tmp/pipeline-requirements.txt \
    && rm -rf /tmp/pip-tmp

FROM devcontainer as runtime

WORKDIR /pipeline
COPY ./pipeline /pipeline

CMD ["run-parts", "--verbose", "."]