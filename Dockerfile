FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.8

WORKDIR /app

COPY ./src/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY ./src /app