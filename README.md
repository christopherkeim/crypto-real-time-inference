[![CI](https://github.com/christopherkeim/crypto-real-time-inference/actions/workflows/cicd.yaml/badge.svg)](https://github.com/christopherkeim/crypto-real-time-inference/actions/workflows/cicd.yaml)
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
![W&B](https://img.shields.io/badge/Experiment%20Tracking-W%26B-yellow?labelColor=black&color=yellow)


# crypto-real-time-inference

The aim of this application is to leverage historical Bitcoin price data and cutting-edge machine learning algorithms to serve inferences about Bitcoin's future price points within a 1 hour window, in real time.

This is my first time working with time series data, and here we will work with raw time series datapoints that are served as OHLC ("open", "high", "low", "close") "candles".

At a high level, I've chosen to think of a given cryptocurrency as a complex system (read: chaotic system), emergent as a phenomenon of large N interactions between groups of humans. Inherently, this is a social system.

In this framework, a "candle" is a measurement of a cryptocurrency's state at a given moment in time - and by measuring state at a series of time points we can see how the system's state evolves over time. The raw dataset itself (consisting of multiple candles) is a function that maps empirically measured states to time points. At a fundamental level, the same concepts can be applied to any physical system composed of a large number of interacting variables - which means this is a very challenging problem!

Here I've chosen to focus on one, and only one, cryptocurrency at the moment: Bitcoin.

> Disclaimer: Cryptocurrency trading involves inherent risks and is subject to market fluctuations. The code here is intended for informational purposes only and should not be considered financial advice. Always conduct thorough research and exercise caution when trading cryptocurrencies.

## Quick Start 🐍 🚀 ✨

### Setup

**Note: for the moment I've targeted Ubuntu 20.04/22.04 for automated setup.**

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
make data
```

### Feature Engineering

5. To build supervised-machine-learning-ready datasets from this raw data, run:

```bash
make features
```

### More to come (see below)

## In Progress 🔧💻

- [x] Continuous Integration (CI)

- [x] Minimum viable data extraction web scraper (CLI tool)

- [x] Minimum viable feature engineering pipeline

- [x] Experiment tracking (Weight & Biases)

- [ ] Training pipeline

- [ ] Inference pipeline (REST API)

- [ ] Frontend

- [ ] Continuous deployment
