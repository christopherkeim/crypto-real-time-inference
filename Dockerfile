# syntax=docker/dockerfile:1


###############################################################
# PYTHON-BASE
# Sets up all our environment variables
###############################################################

FROM python:3.9-slim-buster as python-base

ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # Poetry Version
    POETRY_VERSION=1.5.1 \
    # Set the location that Poetry will install to
    POETRY_HOME="/opt/poetry" \
    # Configure Poetry to create virtual envs in project, '.venv'
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # Non-interactive for automation
    POETRY_NO_INTERACTION=1 \
    # Force Poetry to install at this location
    POETRY_HOME="/opt/poetry" 

# Prepend Poetry to path
ENV PATH="$POETRY_HOME/bin:$PATH"


###############################################################
# BUILDER-BASE
# Used to Build Dependencies + create our virtual environment
###############################################################
FROM python-base as builder-base

# Install system dependencies
RUN apt-get update && apt-get -y install --no-install-recommends \
    ffmpeg \ 
    curl \
    make \
    gcc \
    pciutils

# Install Poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy project pyproject.toml and poetry.lock here to ensure they'll be cached.
COPY . .

# Install runtime dependencies with Poetry - uses $POETRY_VIRTUALENVS_IN_PROJECT and
# $POETRY_NO_INTERACTION 
RUN poetry install


###############################################################
# DEVELOPMENT
# Image used during Development and Testing
###############################################################
FROM builder-base as development

#Flask ENVs

WORKDIR /src

EXPOSE 5000
CMD ["poetry", "run", "ml_server.py"]

#Non-root user
ARG USER="mlspace"