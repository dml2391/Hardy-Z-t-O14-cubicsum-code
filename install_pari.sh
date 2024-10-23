#!/usr/bin/bash

# Ensure the script stops on the first error
set -e

# PARI Version
PARI_VERSION="2.15.4"
#PARI_VERSION="2.17.0"

# File to download
PARI_FILENAME="pari-${PARI_VERSION}.tar.gz"

# Set the URL to download code from - different for the 2.15 family
if [[ $PARI_VERSION == *"2.15"* ]]; then
  PARI_URL="https://pari.math.u-bordeaux.fr/pub/pari/OLD/2.15/${PARI_FILENAME}"
fi

if [[ $PARI_VERSION == *"2.17"* ]]; then
  PARI_URL="https://pari.math.u-bordeaux.fr/pub/pari/unix/${PARI_FILENAME}"
fi

# Script to Download, Compile, and Install PARI

# === Download PARI Source ===
echo "Downloading PARI version ${PARI_VERSION}..."
wget $PARI_URL

# Extract the tarball
echo "Extracting PARI..."
tar zxf $PARI_FILENAME

# Move to the extracted directory
cd "pari-${PARI_VERSION}"

# === Configuration & Installation ===
# Set environment variables required for PARI
export CRAYPE_LINK_TYPE=dynamic

# Configure PARI to be installed in its current directory
echo "Configuring PARI..."
CC=cc ./Configure --prefix=`pwd`

# Compile PARI using 4 cores
echo "Compiling PARI..."
make -j 4 gp

# Run benchmarks to check the installation
echo "Running PARI benchmarks..."
make dobench

# Install PARI
echo "Installing PARI..."
make install

# Get the installation path
PARI_DIR=`pwd`

# Add the PARI directory to .bashrc (if it's not already added)
if ! grep -q "export PARI_DIR=$PARI_DIR" $HOME/.bashrc; then
    echo "Adding PARI paths to .bashrc..."
    echo "# PARI installation paths" >> $HOME/.bashrc
    echo "export PARI_DIR=$PARI_DIR" >> $HOME/.bashrc
    echo "export LD_LIBRARY_PATH=\$PARI_DIR/lib:\$LD_LIBRARY_PATH" >> $HOME/.bashrc
    
    # Source the .bashrc to immediately reflect changes in this script's environment
    source $HOME/.bashrc
fi

# Confirmation message
echo "PARI version ${PARI_VERSION} has been successfully downloaded, compiled, and installed!"

