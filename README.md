[![CI](https://github.com/christopherkeim/python-ml-template/actions/workflows/cicd.yaml/badge.svg)](https://github.com/christopherkeim/python-ml-template/actions/workflows/cicd.yaml)
![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)
![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

# Python Machine Learning Template

This is a template repository for Python-based Machine Learning projects.

## Quick Start 🐍 🚀 ✨

1. Integrated CI pipeline for Python 3.9 and Poetry managed projects (uncomment `make test` when you're ready to have tests in dev or your pipeline)

2. Makefile

3. Dockerfile

4. src package

5. tests package

6. main.py

7. pyproject.toml

8. setup.sh

9. user-story.md Issue Template (for story-driven development)

## Repository Structure

```
├── .github
│     |── ISSUE_TEMPLATE
│     |      └── user-story.md
│     └── workflows
│            └── cicd.yaml
├── src
│    └── __init__.py
|── tests
|     └── __init__.py
|── .gitignore
├── Dockerfile
├── Makefile
├── README.md
├── main.py
├── pyproject.toml
└── setup.sh
```

## Getting Started

This repository is a GitHub Template that you can use to create a new repository for Python-based machine learning projects. It comes pre-configured to use Python3.9 with Poetry 1.5.1 as a package manager.

To get started you can:

1. Click the `Use this template` button at the top right of the page. This will let you create a new repository set up with all of the resources in this template.

2. You can also directly clone this repository:

```bash
git clone https://github.com/christopherkeim/python-template.git
```

## Setup

**Note: for the moment I've targeted Ubuntu 20.04/22.04 development environments for automated setup.**

1. Once you have local copy of this repository in your development environment, navigate into this directory and run the `setup.sh` script:

```bash
cd python-template
bash setup.sh
```

This will install Poetry 1.5.1 and Python3.9 into your development environment.

## Package Management

2. You can configure any dependencies you'd like using the `pyproject.toml` file:

```toml
[tool.poetry.dependencies]
python = ">=3.9, <3.9.7 || >3.9.7, <3.10"

# DevOps
black = "^22.3.0"
click = "^8.1.3"
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
ruff = "^0.0.285"
boto3 = "^1.24.87"

# Web
Flask = "^2.2.2"
Flask-Cors = "^3.0.10"
flask-talisman = "^1.0.0"

# Data Science
jupyter = "^1.0.0"
pandas = "^1.5.0"
numpy = "^1.23.3"
scikit-learn = "^1.1.2"
matplotlib = "^3.6.0"
seaborn = "^0.12.0"
```

## Package Installation

3. Once you're happy with your defined dependencies, you can run `make install` (or `poetry install` directly) to install the Python dependencies for your project into a virtual environment (pre-configured to be placed in your project's directory):

```bash
make install
```

4. This will create a `poetry.lock` file defining exactly what dependencies you're using in development and testing. It's recommended that you check this file into version control so others can recreate this on their machines 💻 and in production 🚀.

## Fire Up Some Code!

5. You're all set to start developing 🐍 🚀 ✨.

## Continuous Integration

You'll want to edit the `README.md` and replace the CI badge with a hook for your specific repository's GitHub Actions CI workflow.

## Continuous Delivery

You can also add a deploy target by editing your `Makefile` or the `cicd.yaml` GitHub Actions workflow file.

## Why Poetry?

As I'm learning more about DevOps and the joys of dependency management in Python projects, I've noticed that Software Engineering and MLOps minded folks tend to like Poetry. There's a few reasons I think Poetry is a solid choice for setting your code up to survive across different environments at the level of dependency management:

1. It allows you to express what primary dependencies you believe your application will work with using the `pyproject.toml` file, and allow for upgrade paths down the road

2. Unlike `pip`, the `poetry.lock` file lets you define exactly what dependencies you're using in development and testing. This means your Python dependency structure can be exactly replicated on other machines, every time.

3. Poetry has very convenient virtual environment management (which we've configured here to be placed within your project directory)
