FROM python:3.10.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libegl1-mesa \
    libgl1-mesa-dri \
    mesa-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install  -r requirements.txt

COPY . /app/

EXPOSE 8501

CMD [ "python" , "-m", "streamlit", "run", "app.py"]
