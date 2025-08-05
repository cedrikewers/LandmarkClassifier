FROM python:3.12-slim

WORKDIR /app

# Runtime library for uwsgi
RUN apt update && apt install libexpat1

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN mkdir models label_samples && \
    tar -xzf models.tar.gz -C models && \
    tar -xzf label_samples.tar.gz -C label_samples

EXPOSE 5025
CMD ["uwsgi", "uwsgi.ini"]
