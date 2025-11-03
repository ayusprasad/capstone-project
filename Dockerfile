FROM python:3.11-slim

WORKDIR /app

# Copy Flask app and requirements
COPY flask_app/ /app/

# Create models directory
RUN mkdir -p /app/models

# Copy model files 
COPY models/* /app/models/

# Ensure correct permissions
RUN chmod -R 755 /app/models

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]