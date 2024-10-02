#!/usr/bin/bash

# Ensure the script stops on the first error
set -e

# Ensure PARI_DIR is set (from the installation script)
if [ -z "$PARI_DIR" ]; then
    echo "Error: PARI_DIR is not set. Please make sure you have sourced your .bashrc or installed PARI using the provided script."
    exit 1
fi

# === Compile the Fortran Program using the PARI Library ===

# Explanation of the compile command components:
# ftn               : Fortran compiler command
# -L$PARI_DIR/lib/  : Specifies the directory where the compiler can find the PARI library using the PARI_DIR variable.
# -lpari            : Links the PARI library. The "l" prefix tells the compiler to link against the library named "pari".
# zeta.f90          : The Fortran source file to be compiled.
# -o zeta.out       : Specifies the output name for the compiled executable.

# Actual compile command
ftn -O3 -L$PARI_DIR/lib/ -lpari zeta.f90 -o zeta.out

# Confirmation message
echo "Compilation successful! Executable generated as 'zeta.out'."

