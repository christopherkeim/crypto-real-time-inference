#! /bin/bash
# This script sets up a deployment environment on an Ubuntu 20.04/22.04 OR Debian
# machine to work with Poetry managed Python3.10 source code.
# 
# Targets:
#   - Docker 24.0.6
#   - Go1.21.1
#
# Requirements:
#   - Ubuntu 20.04/22.04
#   - Python3.7+ 
#
# -----------------------------------------------------------------------------------------------------------
# 1) Base Requirements: this will ensure that you have ca-certificates, curl, make, and gnupg installed.
# -----------------------------------------------------------------------------------------------------------

# Check if ca-certificates is in the apt-cache
if ( apt-cache show ca-certificates > /dev/null )
then
  echo "ca-certificates is already cached 🟢"
else
  sudo apt update
fi

# Ensure ca-certificates package is installed on the machine
if ( which update-ca-certificates > /dev/null )
then
  echo "ca-certificates is already installed 🟢"
else
  echo "Installing ca-certificates 📜"
  sudo apt-get install -y ca-certificates
fi

# Check if curl is in the apt-cache
if ( apt-cache show curl > /dev/null )
then
  echo "curl is already cached 🟢"
else
  sudo apt update
fi

# Ensure curl is installed on the machine
if ( which curl > /dev/null )
then
  echo "curl is already installed 🟢"
else
  echo "Installing curl 🌀"
  sudo apt install -y curl
fi

# Check if wget is in the apt-cache
if ( apt-cache show wget > /dev/null )
then
  echo "wget is already cached 🟢"
else
  sudo apt update
fi

# Ensure wget is installed on the machine
if ( which wget > /dev/null )
then
  echo "wget is already installed 🟢"
else
  echo "Installing wget 🌀"
  sudo apt install -y wget
fi

# Check if make is in the apt-cache
if ( apt-cache show make > /dev/null )
then
  echo "make is already cached 🟢"
else
  sudo apt update
fi

# Ensure make is installed on the machine
if ( which make > /dev/null )
then
  echo "make is already installed 🟢"
else
  echo "Installing make 🔧"
  sudo apt install -y make
fi

# Check if gnupg is in the apt-cache
if ( apt-cache show gpg > /dev/null )
then
  echo "gnupg is already cached 🟢"
else
  sudo apt update
fi

# Ensure gnupg is installed on the machine
if ( which gpg > /dev/null )
then
  echo "make is already installed 🟢"
else
  echo "Installing gnugp 🔧"
  sudo apt install -y gnupg
fi


# -----------------------------------------------------------------------------------------------------------
# 2) Docker Install: here we'll install Docker
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
# 2.1) Set up the repository: Before you install Docker Engine for the first time on a new host machine, 
# you need to set up the Docker repository. Afterward, you can install and update Docker from the repository.
# -----------------------------------------------------------------------------------------------------------

# Pull the current machine's distro for GPG key targeting
DISTRO=$(lsb_release -d | awk -F ' ' '{print tolower($2)}')

# Add Docker’s official GPG key
if [ -f /etc/apt/keyrings/docker.gpg ]
then
  echo 'Docker GPG Key already installed at /etc/apt/keyrings/docker.gpg 🟢'
else
  echo 'Installing Docker GPG Key at /etc/apt/keyrings/docker.gpg 🔧'
  
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
# 2.2) Install Docker Engine (with Docker Compose)
# -----------------------------------------------------------------------------------------------------------

# Check if docker-ce is in the apt-cache
if ( apt-cache show docker-ce > /dev/null )
then
  echo "docker-ce is already cached 🟢"
else
  sudo apt update
fi

# Install Docker Engine, containerd, and Docker Compose
if ( docker --version > /dev/null )
then
  echo "Docker is already installed 🟢"
  echo "Using $(docker --version)"
else
  echo "Installing Docker 🐳"

  # Installs
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  
  # Verify that the Docker Engine installation is successful by running the hello-world image
  sudo docker run --rm hello-world
fi



# -----------------------------------------------------------------------------------------------------------
# 3) Go Install: here we'll install and configure Go
# -----------------------------------------------------------------------------------------------------------

# Pull the machine's chip architecture
if [ "$(uname -m)" == "x86_64" ]
then
  CHIP_ARCH="amd64"
else
  CHIP_ARCH="arm64"
fi

# Install Go
if ( which go > /dev/null )
then
  echo "Go is already installed 🟢"
else
  echo "Installing Go 🦫"
  wget -O https://go.dev/dl/go1.21.3.linux-$CHIP_ARCH.tar.gz
  sudo tar -C /usr/local -xzf go1.21.3.linux-$CHIP_ARCH.tar.gz
  echo -e "# Add Go to PATH\nexport PATH="/usr/local/go/bin:$PATH"" >> ~/.profile
  source ~/.profile
fi

# Verify installation of Go
if ( go version > /dev/null )
then
  echo "$(go version) 🦫 🚀"
else
  echo "Go was not installed successfully 🔴"
fi


# -----------------------------------------------------------------------------------------------------------
# 4) Build the Go Webhook Server
# -----------------------------------------------------------------------------------------------------------

# Build the webhook server if it does not exist
if [ -f deploy/webhook ]
then
  echo "webhook already built 🟢"
else
  echo "Building webhook 🦫"
  pushd deploy
  go build webhook.go
  popd
fi

# -----------------------------------------------------------------------------------------------------------
# 5) Add execute permissions to `deploy_container.sh`
# -----------------------------------------------------------------------------------------------------------

if [ "$(ls -l deploy/deploy_container.sh | cut -d " " -f1)" == "-rwxrw-r--" ]
then
  echo "Permissions set for deploy_container.sh 🟢"
else
  echo "Setting permissions for deploy_container.sh 🐳"
  chmod u+x deploy/deploy_container.sh
fi

# Verify permissions for deploy_container.sh
if [ "$(ls -l deploy/deploy_container.sh | cut -d " " -f1)" == "-rwxrw-r--" ]
then
  echo "Permissions set for deploy_container.sh 🟢 🐳"
else
  echo "Error setting permissions for deploy_container.sh 🔴"
fi
