#! /bin/bash
# This script sets up a deployment environment on an Ubuntu 20.04/22.04 OR Debian
# machine to work with Poetry managed Python3.10 source code.
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
# 2) Poetry Install: here we'll install and configure Poetry, as well as add Poetry to the PATH.
# -----------------------------------------------------------------------------------------------------------

# Install Poetry using the official installer
if ( which poetry > /dev/null )
then
  echo "Poetry is already installed 🟢"
else
  echo "Installing Poetry 🧙‍♂️"
  curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.5.1 python3 -
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
if ( which python3.10 > /dev/null )
then
  echo "Python3.10 already installed 🐍"
else
  echo "Installing Python3.10 🔧"
  sudo apt install -y python3.10
  sudo apt-get install -y gcc python3.10-dev
fi

# Verify Python3.10 installation
if ( which python3.10 > /dev/null )
then
  echo "$(python3.10 --version) 🐍 🚀 ✨"
else
  echo "Python 3.10 was not installed successfully 🔴"
fi

# -----------------------------------------------------------------------------------------------------------
# 3) Python Dependency Installation: this will install all source code dependencies defined in the 
#    `poetry.lock` file onto the machine
# -----------------------------------------------------------------------------------------------------------
poetry install