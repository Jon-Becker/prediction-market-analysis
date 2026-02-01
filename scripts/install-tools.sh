#!/bin/bash
set -e

TOOLS=("zstd" "aria2c")

# Check if a tool is installed
check_tool() {
    local tool=$1
    if command -v "$tool" &> /dev/null; then
        echo "$tool is already installed"
        return 0
    fi
    return 1
}

# Get package name for a tool (may differ from binary name)
get_package_name() {
    local tool=$1
    case "$tool" in
        aria2c)
            echo "aria2"
            ;;
        *)
            echo "$tool"
            ;;
    esac
}

# Detect OS and return install command for a package
get_install_command() {
    local package=$1
    case "$(uname -s)" in
        Darwin)
            echo "brew install $package"
            ;;
        Linux)
            if command -v apt-get &> /dev/null; then
                echo "sudo apt-get install -y $package"
            elif command -v dnf &> /dev/null; then
                echo "sudo dnf install -y $package"
            elif command -v yum &> /dev/null; then
                echo "sudo yum install -y $package"
            elif command -v pacman &> /dev/null; then
                echo "sudo pacman -S --noconfirm $package"
            else
                echo ""
            fi
            ;;
        *)
            echo ""
            ;;
    esac
}

# Install a tool
install_tool() {
    local tool=$1
    local package
    package=$(get_package_name "$tool")
    local install_cmd
    install_cmd=$(get_install_command "$package")

    if [ -z "$install_cmd" ]; then
        echo "Error: Unable to determine package manager for this OS."
        echo "Please install $tool manually."
        exit 1
    fi

    echo "$tool is not installed."
    echo "Install command: $install_cmd"
    echo ""
    read -p "Do you want to install $tool? [y/N] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing $tool..."
        eval "$install_cmd"
        echo "$tool installed successfully."
    else
        echo "Installation cancelled."
        exit 1
    fi
}

# Main
for tool in "${TOOLS[@]}"; do
    if ! check_tool "$tool"; then
        install_tool "$tool"
    fi
done
