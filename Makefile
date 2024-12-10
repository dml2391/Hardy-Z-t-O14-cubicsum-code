# Makefile to compile code
#
# Choose the particular source you would like to compile by
# commenting/uncommenting the particular file you want to compile.
# It assumes that you have compiled the PARI library and it is 
# available and that the PARI_DIR environment variable points to it.

# Define variables, see stackoverflow 4879592 for different type of assignments.

# Disable make built in rules and variables to avoid unexpected behaviour.
MAKEFLAGS += --no-builtin-rules --no-builtin-variables

# Fortran compiler to be used to compile the code.
FC  := ftn # Cray compiler wrapper - need the GNU compilers.
#FC  := mpif90 # MPI compiler

# Choose the source file(s) to compile by uncommenting the version you 
# want to use and commenting out the ones that you do not want.

#SRC := src/zeta14cubicmult.f90
SRC := src/zeta14cubicmultharness.f90
#SRC := src/zeta15quartic.f90

# Libraries to link against. Make sure that the root location of the
# PARI library is set in PARI_DIR.
LIB := -L$(PARI_DIR)/lib/ -lpari

# Command to remove files - the backslash removes any aliasing.
RM := \rm -f

# Name of the executable to be produced.
EXE := zeta_dml.out

# Compiler flags (you may need to change some of these):
#
# -Wall enables common compiler warning options.
# -Wextra enables warning options for usage of language features that may be problematic.
# -fcheck=all enable all run time checks
# -z noexecstack silences the executable stack warning.
#
FFLAGS := -O3 -z noexecstack -fallow-argument-mismatch #-Wall -Wextra #-fcheck=all

# Create list of object files by substituting f90 with an .o extension.
OBJ := $(SRC:.f90=.o)

# Rebuild object file if the Makefile changes.
$(OBJ): $(MAKEFILE_LIST)

# The default target.
.DEFAULT_GOAL := all

# Targets
.PHONY: all clean pat

# The default target
all: $(EXE)

# Rule to link the object files into the executable.
# $@ - target (the bit to the left of the colon)
# $^ - dependencies (the bit to the right of the colon)
$(EXE): $(OBJ) 
	$(FC) $(FFLAGS) -o $@ $^ $(LIB)

# Rule to go from a .o from a .f90. The "%" is a wild card.
%.o : %.f90
	$(FC) $(FFLAGS) -c -o $@ $<

# Make an instrumented version of the code for profiling on a cray system
pat: $(EXE)
	pat_build  -O apa $^

# Remove derived and other files. Backslash is used to remove any aliases for rm.
clean:
	$(RM)  *.o src/*.o zeta_dml.out zeta_dml.out+pat *~ src/*~


