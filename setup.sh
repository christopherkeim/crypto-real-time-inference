#! /bin/bash
# This script sets up a local development environment on an Ubuntu 20.04/22.04 machine
# to work with Poetry managed Python3.10 projects. 
# 
# Targets:
#   - Poetry 1.5.1
#   - Python3.10
#
# Requirements:
#   - Ubuntu 20.04/22.04
#   - Python3.7+ 
#
# -----------------------------------------------------------------------------------------------------------
# 1) Base Requirements: this will ensure that you have curl and make installed.
# -----------------------------------------------------------------------------------------------------------

# Check if curl is in the apt-cache
if ( apt-cache show curl > /dev/null )
then
  echo "curl is already cached 🟢"
else
  sudo apt update
fi

# Ensure curl is installed on the machine
if [ -n "$(which curl)" ]
then
  echo "curl is already installed 🟢"
else
  echo "Installing curl 🌀"
  sudo apt install -y curl
fi

# Check if make is in the apt-cache
if ( apt-cache show make > /dev/null )
then
  echo "make is already cached 🟢"
else
  sudo apt update
fi

# Ensure make is installed on the machine
if [ -n "$(which make)" ]
then
  echo "make is already installed 🟢"
else
  echo "Installing make 🔧"
  sudo apt install -y make
fi

# -----------------------------------------------------------------------------------------------------------
# 2) Poetry Install: here we'll install and configure Poetry, as well as add Poetry to the PATH.
# -----------------------------------------------------------------------------------------------------------

# Install Poetry using the official installer
if [ -n "$(which poetry)" ]
then
  echo "Poetry is already installed 🟢"
else
  echo "Installing Poetry 🧙‍♂️"
  curl -sSL https://install.python-poetry.org | python3 -
fi

# Add Poetry to the path in the current user's .bashrc
if ( poetry --version > /dev/null )
then
  echo "Poetry is already in PATH 🟢"
else
  echo -e "# Add Poetry (Python Package Manager) to PATH\nexport PATH="/home/$USER/.local/bin:$PATH"" >> ~/.bashrc
  source ~/.bashrc
fi

# Configure Poetry to put build all virtual environments in the project's directory
if [ "$(poetry config virtualenvs.in-project)" == "true" ]
then
  echo "Poetry already configured to create virtual envs within projects 🟢"
else
  echo "Configuring Poetry to create virtual envs in projects 🪐"
  poetry config virtualenvs.in-project true
fi

# -----------------------------------------------------------------------------------------------------------
# 3) Python3.10 Install: here we'll install Python3.10 - feel free to swap this for any version you'd like.
# -----------------------------------------------------------------------------------------------------------

# Check if software-properties-common is in the apt-cache
if ( apt-cache show software-properties-common > /dev/null )
then
  echo "software-properties-common is already cached 🟢"
else
  sudo apt update
fi

# Check for the software-properties-common requirement
if ( dpkg -L software-properties-common > /dev/null )
then
  echo "software-properties-common requirement met 🟢"
else
  echo "Installing software-properties-common 🔧"
  sudo apt install -y software-properties-common
fi

# Add this apt repository for Python 3.9
if [ -n "$(ls /etc/apt/sources.list.d | grep deadsnakes)" ]
then
  echo "ppa:deadsnakes/ppa apt repository present 🟢"
else
  echo "Adding deadsnakes to the apt-repository 💀🐍"
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  # Refresh the package list again
  sudo apt update
fi

# Now you can download Python3.10
if [ -n "$(which python3.10)" ]
then
  echo "Python3.10 already installed 🐍"
else
  echo "Installing Python3.10 🔧"
  sudo apt install -y python3.10
fi

# Verify Python3.10 installation
if [ -n "$(which python3.10)" ]
then
  echo "$(python3.10 --version) 🐍 🚀 ✨"
else
  echo "Python 3.10 was not installed successfully 🔴"
fi

# -----------------------------------------------------------------------------------------------------------
# 4) Docker Install: here we'll install Docker
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
# 4.1) Set up the repository: Before you install Docker Engine for the first time on a new host machine, 
# you need to set up the Docker repository. Afterward, you can install and update Docker from the repository.
# -----------------------------------------------------------------------------------------------------------

DISTRO=$(lsb_release -d | awk -F ' ' '{print tolower($2)}')

# Add Docker’s official GPG key
if [ -f /etc/apt/keyrings/docker.gpg ]
then
  echo 'Docker GPG Key already installed at /etc/apt/keyrings/docker.gpg 🟢'
else
  echo 'Installing Docker GPG Key at /etc/apt/keyrings/docker.gpg 🔧'
  
  # Update the apt package index and install packages to allow apt to use a repository over HTTPS
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  
  # Create the /etc/apt/keyrings directory with appropriate permissions
  sudo install -m 0755 -d /etc/apt/keyrings
  
  # Download the GPG key from Docker
  curl -fsSL https://download.docker.com/linux/$DISTRO/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
fi

# Set up the repository
if [ -f /etc/apt/sources.list.d/docker.list ] 
then
  echo 'docker.list repository already exists at /etc/apt/sources.list.d/docker.list 🟢'
else
  echo 'Installing docker.list repository at /etc/apt/sources.list.d/docker.list 🔧'
  echo \
    "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$DISTRO \
    "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
fi

# -----------------------------------------------------------------------------------------------------------
# 4.2) Install Docker Engine
# -----------------------------------------------------------------------------------------------------------

# Check if docker-ce is in the apt-cache
if ( apt-cache show docker-ce > /dev/null )
then
  echo "docker-ce is already cached 🟢"
else
  sudo apt update
fi

# Install Docker Engine, containerd, and Docker Compose
if [ "$(docker --version)" ]
then
  echo "Docker is already installed 🟢"
  echo "Using $(docker --version)"
else
  echo "Installing Docker 🐳"

  # Installs
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  
  # Verify that the Docker Engine installation is successful by running the hello-world image
  sudo docker run hello-world
fi