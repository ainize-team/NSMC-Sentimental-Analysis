FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

WORKDIR /app

COPY ./src/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY ./src /app
