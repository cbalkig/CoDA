#!/bin/bash

OS_TYPE=$(uname)
echo "🚀 Starting unified setup script for $OS_TYPE..."

# Check if command exists
command_exists() {
    command -v "$1" &>/dev/null
}

has_gpu() {
    lspci | grep -i nvidia &>/dev/null
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT" || { echo "Failed to navigate to project root"; exit 1; }

# Install pyenv prerequisites (Linux)
if [[ "$OS_TYPE" == "Linux" ]]; then
    sudo apt update
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl vim

    sudo apt-get update
    sudo apt-get install -y linux-headers-$(uname -r)

    sudo apt-get install -y wget gnupg
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update

    # Install a recent datacenter driver (R550 series) and CUDA 12.4 toolkit
    sudo apt-get install -y nvidia-driver-550
    sudo apt-get install -y cuda-toolkit-12-4

    # Set paths (do this once; avoid duplicates)
    if ! grep -q '/usr/local/cuda/bin' ~/.bashrc; then
      echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
      echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi

    # Install pyenv
    if ! command_exists pyenv; then
        curl https://pyenv.run | bash
        echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        source ~/.bashrc
    fi

elif [[ "$OS_TYPE" == "Darwin" ]]; then
    # Install Homebrew if necessary
    if ! command_exists brew; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    brew update
    brew install pyenv

    # Add pyenv to path for macOS
    CURRENT_SHELL=$(basename "$SHELL")
    if [[ "$CURRENT_SHELL" == "zsh" ]]; then
        SHELL_PROFILE="$HOME/.zshrc"
    else
        SHELL_PROFILE="$HOME/.bash_profile"
    fi

    grep -q 'pyenv init' "$SHELL_PROFILE" || {
        echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> "$SHELL_PROFILE"
        echo 'eval "$(pyenv init --path)"' >> "$SHELL_PROFILE"
        echo 'eval "$(pyenv init -)"' >> "$SHELL_PROFILE"
        source "$SHELL_PROFILE"
    }
fi

# Install Python 3.10 via pyenv if not present
if ! pyenv versions | grep -q 3.10.14; then
    pyenv install 3.10.14
fi
pyenv global 3.10.14

# Common setup after Python installation
PYTHON_BIN=$(pyenv which python)

# Bootstrap .env from the template if the user hasn't created one yet.
# Real credentials must NEVER be committed — .env is gitignored.
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    if [[ -f "$PROJECT_ROOT/.env.template" ]]; then
        cp "$PROJECT_ROOT/.env.template" "$PROJECT_ROOT/.env"
        # Stamp the resolved Python interpreter so other scripts can find it.
        if [[ "$OS_TYPE" == "Darwin" ]]; then
            sed -i '' "s|^PYTHON_BIN=.*|PYTHON_BIN=$PYTHON_BIN|" "$PROJECT_ROOT/.env"
        else
            sed -i "s|^PYTHON_BIN=.*|PYTHON_BIN=$PYTHON_BIN|" "$PROJECT_ROOT/.env"
        fi
        echo "✅ .env created from .env.template — fill in GITHUB_USERNAME/TOKEN/REPO before pushing."
    else
        echo "⚠️  No .env.template found; skipping .env creation."
    fi
else
    echo "ℹ️  .env already exists; leaving it untouched."
fi

# Setup Python environment
$PYTHON_BIN -m pip install --upgrade pip setuptools

# Create and activate virtual environment
$PYTHON_BIN -m venv "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"

# Git configuration is left to the user.
# Set your own with:
#   git config --global user.email "you@example.com"
#   git config --global user.name "Your Name"

rm -rf ~/.local/share/Trash/*

"$PROJECT_ROOT/.venv" -m pip install -r requirements.txt

echo "🎉 Setup completed successfully!"

[[ "$OS_TYPE" == "Linux" ]] && sudo apt autoremove -y
[[ "$OS_TYPE" == "Darwin" ]] && brew cleanup

git config --global credential.helper store
git pull
