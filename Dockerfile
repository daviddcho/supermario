FROM python:3.8

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsdl2-dev \
    swig \
    cmake \
    x11-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libgles2-mesa-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Entry point
#CMD ["python", "mario.py", "play"]
