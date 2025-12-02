#!/bin/bash

# SoftFloat Installation Script for FPY Benchmarks
# This script downloads and builds Berkeley SoftFloat 3e

set -e  # Exit on any error

SOFTFLOAT_VERSION="3e"
SOFTFLOAT_DIR="SoftFloat-${SOFTFLOAT_VERSION}"
INSTALL_DIR="$(pwd)/third_party/softfloat"

echo "=============================================="
echo "Installing Berkeley SoftFloat ${SOFTFLOAT_VERSION}"
echo "=============================================="

# Create third_party directory if it doesn't exist
mkdir -p third_party
cd third_party

# Download SoftFloat if not already present
if [ ! -f "${SOFTFLOAT_DIR}.zip" ]; then
    echo "Downloading SoftFloat ${SOFTFLOAT_VERSION}..."
    wget "http://www.jhauser.us/arithmetic/${SOFTFLOAT_DIR}.zip"
else
    echo "SoftFloat archive already exists, using cached version."
fi

# Extract if directory doesn't exist
if [ ! -d "$SOFTFLOAT_DIR" ]; then
    echo "Extracting SoftFloat..."
    unzip "${SOFTFLOAT_DIR}.zip"
else
    echo "SoftFloat directory already exists."
fi

cd "$SOFTFLOAT_DIR"

# Determine build target based on system
UNAME_M=$(uname -m)
UNAME_S=$(uname -s)

if [ "$UNAME_S" = "Linux" ]; then
    if [ "$UNAME_M" = "x86_64" ]; then
        BUILD_TARGET="Linux-x86_64-GCC"
    elif [ "$UNAME_M" = "aarch64" ]; then
        BUILD_TARGET="Linux-ARM-VFPv2-GCC"  # Best available ARM target
    else
        BUILD_TARGET="Linux-386-GCC"  # Fallback
    fi
elif [ "$UNAME_S" = "Darwin" ]; then
    BUILD_TARGET="Linux-x86_64-GCC"  # Use Linux target as fallback
else
    BUILD_TARGET="Linux-x86_64-GCC"  # Default fallback
fi

echo "Using build target: $BUILD_TARGET"

# Build SoftFloat
cd "build/$BUILD_TARGET"

echo "Building SoftFloat..."
make clean 2>/dev/null || true  # Clean if possible
make

# Create installation directory structure
echo "Installing SoftFloat to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR/lib"
mkdir -p "$INSTALL_DIR/include"

# Copy library
cp softfloat.a "$INSTALL_DIR/lib/libsoftfloat.a"

# Copy headers
cd ../../source/include
cp softfloat.h "$INSTALL_DIR/include/"
cp softfloat_types.h "$INSTALL_DIR/include/" 2>/dev/null || true

echo "=============================================="
echo "SoftFloat installation completed!"
echo "=============================================="
echo "Library: $INSTALL_DIR/lib/libsoftfloat.a"
echo "Headers: $INSTALL_DIR/include/softfloat.h"
echo ""
echo "To use SoftFloat in your builds, add:"
echo "  -I$INSTALL_DIR/include"
echo "  -L$INSTALL_DIR/lib -lsoftfloat"
echo ""
echo "Now you can uncomment the SoftFloat code in:"
echo "  benchmark_sqrt_with_softfloat.cpp"
echo "=============================================="
