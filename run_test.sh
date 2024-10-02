if [ ! -f zeta14v3resa ]; then
    touch zeta14v3resa
fi

# Run the compiled Fortran program 'zeta.out' in parallel
# Using 'time' to measure its execution duration
./zeta.out

