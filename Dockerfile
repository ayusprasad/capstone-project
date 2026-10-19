FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

COPY flask_app/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && mkdir -p "$NLTK_DATA" \
    && python -m nltk.downloader -d "$NLTK_DATA" stopwords wordnet

COPY flask_app/ ./
COPY models/ ./models/
COPY reports/ ./reports/

RUN adduser --disabled-password --no-create-home app \
    && chown -R app:app /app

USER app

EXPOSE 5000
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "app:app"]
