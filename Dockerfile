FROM python:3.12-slim-bullseye AS devcontainer

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get -y install curl wget git

COPY ./requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

CMD ["/bin/bash"]
# FROM devcontainer AS deploy

# WORKDIR /lcars
# COPY lcars /lcars

# # Download and extract the Memory Alpha RAG database from the specified URL
# RUN wget $MEMORY_ALPHA_RAG_DB_URL -O /tmp/memoryalpha.tar.gz && \
#     tar -xzf /tmp/memoryalpha.tar.gz -C /data && \
#     rm /tmp/memoryalpha.tar.gz