FROM ubuntu:latest

RUN apt update -y
RUN apt install -y python-pip python-dev build-essential

COPY . /app
WORKDIR /app


ENTRYPOINT ["python"]