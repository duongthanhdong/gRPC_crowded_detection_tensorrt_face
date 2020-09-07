FROM docker-registry.vnpttiengiang.vn/crowded/docker-base-tensorrt6-cuda10.1-cudnn7.6:latest

ENV APP_DIR /app
ENV PYTHONUNBUFFERED TRUE
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR ${APP_DIR}

COPY ./requirements.txt ${APP_DIR}
RUN pip3 install -r requirements.txt

COPY . ${APP_DIR}
EXPOSE 30055

#WORKDIR ${APP_DIR}/usage_tensorrt_model/code_to_create_model

RUN bash /app/usage_tensorrt_model/code_to_create_model/create_tensorRT_model.sh

#WORKDIR ${APP_DIR}
