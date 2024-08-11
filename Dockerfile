FROM python:3.12.3-slim

WORKDIR /app

# Install Node.js
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs
# Install mineflayer
RUN npm install mineflayer

COPY requirements.txt /app/

RUN pip3 install -r requirements.txt

RUN apt-get update && \
    apt-get install openscad -y

COPY keys.json /app/
COPY config.json /app/
# COPY build_prompt_description.txt /app/
COPY request_to_scad.py /app/
COPY openscad_to_voxels.py /app/
COPY builder_system_prompt.txt /app/
COPY bot.py /app/bot.py

CMD ["python3", "bot.py"]