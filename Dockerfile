FROM　ubuntu:latest
COPY . /app/source_code/
RUN mkdir -p /app/tianji && cp -r /app/source_code/* /app/tianji
WORKDIR /app/tianji
