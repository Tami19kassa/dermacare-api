# Step 1: Specify the base image.
# We are using an official, specific, and stable version of Python.
# The 'slim-buster' tag provides a lightweight Debian-based Linux environment.
FROM python:3.11-slim-buster

# Step 2: Set the working directory inside the container.
# This is where our application's code will live.
WORKDIR /app

# Step 3: Copy the dependencies file into the container.
# We copy this file first to take advantage of Docker's layer caching.
# If requirements.txt doesn't change, Docker won't re-run the installation on subsequent builds.
COPY requirements.txt .

# Step 4: Install the Python dependencies.
# This command runs on the clean Linux server inside the container,
# which has all the necessary compilers and tools, avoiding all local machine issues.
# --no-cache-dir is a good practice to keep the image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application's source code.
# This includes your main.py and the assets folder with your model.
COPY . .

# Step 6: Define the command to run your application.
# Gunicorn is a professional-grade production server for Python web apps.
# 'app:app' means "in the file named app.py, find the Flask object named app".
# We bind to port 8080, which is a common port for web services.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]