[![CI](https://github.com/christopherkeim/crypto-real-time-inference/actions/workflows/ci.yaml/badge.svg)](https://github.com/christopherkeim/crypto-real-time-inference/actions/workflows/ci.yaml)
[![Build and Push Prediction Service to Docker Hub](https://github.com/christopherkeim/crypto-real-time-inference/actions/workflows/build-and-push-to-docker-hub.yaml/badge.svg)](https://github.com/christopherkeim/crypto-real-time-inference/actions/workflows/build-and-push-to-docker-hub.yaml)
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
![W&B](https://img.shields.io/badge/Experiment%20Tracking-W%26B-yellow?labelColor=black&color=yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Next JS](https://img.shields.io/badge/Next-black?style=flat&logo=next.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=flat&logo=typescript&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=flat&logo=tailwind-css&logoColor=white)

# crypto-real-time-inference

The aim of this application is to leverage historical Bitcoin price data and cutting-edge machine learning algorithms to serve inferences about Bitcoin's future price points within a 1 hour window, in real time.

This is my first time working with time series data, and here we will work with raw time series datapoints that are served as OHLC ("open", "high", "low", "close") "candles".

At a high level, I've chosen to think of a given cryptocurrency as a complex system (read: chaotic system), emergent as a phenomenon of large N interactions between groups of humans. Inherently, this is a social system.

In this framework, a "candle" is a measurement of a cryptocurrency's state at a given moment in time - and by measuring state at a series of time points we can see how the system's state evolves over time. The raw dataset itself (consisting of multiple candles) is a function that maps empirically measured states to time points. At a fundamental level, the same concepts can be applied to any physical system composed of a large number of interacting variables - which means this is a very challenging problem!

Here I've chosen to focus on one, and only one, cryptocurrency at the moment: Bitcoin.

## Disclaimer

**Cryptocurrency trading involves inherent risks and is subject to market fluctuations. The code here is intended for informational purposes only and should not be considered financial advice. Always conduct thorough research and exercise caution when trading cryptocurrencies.**

## Quick Start 🐍 🚀 ✨

### Setup

**Note: for the moment I've targeted Ubuntu 20.04/22.04 for automated dev setup.**

1. You can clone this repository onto a machine with:

```bash
git clone https://github.com/christopherkeim/crypto-real-time-inference.git
```

2. Once you have local copy of this repository, navigate into this directory and run the `setup.sh` script:

```bash
cd crypto-real-time-inference
bash setup.sh
```

This will install Poetry 1.5.1 and Python3.10 into your environment.

### Dependency Installation

3. To install the Python dependencies for this application, run:

```bash
make install
```

### Data

4. To download Bitcoin candles using default parameters (from September 2020 - September 2023) run:

```bash
make rawdata
```

### Feature Engineering

5. To build supervised-machine-learning-ready datasets from this raw data, run:

```bash
make features
```

### Machine Learning Training

6. To build a Lasso Regressor model (primary recommendation) using default parameters, run:

```bash
make train
```

### Deep Learning Training

7. To build a Convolutional Neural Network (primary recommendation) using default parameters, run:

```bash
make nntrain
```

### Model Prediction (Endpoint)

8. To start the prediction service locally using FastAPI and Uvicorn, run:

```bash
make predict
```

You can curl the `http://0.0.0.0:8000/api/predict` endpoint or simply navigate to that URL in your browser to garner predictions from your trained Convolutional Neural Network for the current next hour's price point (defaults)

## Prediction Backend 🧙‍♂️ 🔧

### Prediction Service Containerization

1. To build the prediction service into a Docker container, navigate to the root of this repository and run:

```bash
docker build -t crypto-real-time-inference:v0 .
```

2. To start prediction service container, run:

```bash
docker run -p 8000:8000 crypto-real-time-inference:v0
```

The containerized prediction service will serve predictions at `http://0.0.0.0:8000/api/predict`.

## Frontend 🪅 ✨

### Frontend Client Setup

1. To setup the frontend client, navigate to the `frontend` directory and run:

```bash
npm install
```

You will also need to make a copy of `.env.local.example` and rename it to `.env.local`:

```bash
cp .env.local.example .env.local
```

The default values in `.env.local` should work out of the box for local development. If you change where the backend prediction service is hosted, you will need to update the `CRYPTO_INFERENCE_API_URI` variable in `.env.local` to reflect the new URI.

### Frontend Client Development

2. To start the frontend client development server, navigate to the `frontend` directory and run:

```bash
npm run dev
```

The default configuration will spin up the frontend client development server at `http://localhost:3000` and the backend prediction service at `http://localhost:8000`, with Hot Module Reload enabled for both.

It is also possible to run the frontend sever by itself, without the backend prediction service, by running:

```bash
npm run next-dev
```

### Frontend Client Production Build

3. To build the frontend client for production, navigate to the `frontend` directory and run:

```bash
npm run build
```

This will build the frontend client into the `frontend/.next` directory. To serve the production build, run:

```bash
npm run start
```

A great candidate for deployment is [Vercel](https://vercel.com), just make sure you set the `frontend` directory as the project directory after linking your repo. Other cloud providers will work as long as then call `npm run build` and `npm run start` in the root of the `frontend` directory.

### More to come (see below)

## In Progress 🔧💻

- [x] Continuous Integration (CI)

- [x] Minimum viable Data Extraction Web Scraper (CLI tool)

- [x] Minimum viable Feature Engineering Pipeline

- [x] Experiment tracking (Weight & Biases)

- [x] Minimum viable Training Pipelines (ML & DL)

- [x] Minimum viable Inference Pipeline (REST API)

- [x] Continuous Delivery to Docker Hub (CD, `x86_64`, `arm64` targets)

- [ ] Frontend

- [ ] Continuous deployment
