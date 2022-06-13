# start by pulling the python image
FROM python:3.8
ENV PYTHONUNBUFFERED 1

# create app directory
WORKDIR /app

# Copy requirement file to directory app
COPY requirements.txt /app/

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app/

# expore container port
EXPOSE 8080

# run service
CMD exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 0 app:app