# Hardy-Z-t-code
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A repository of Fortran/C codes for the rapid calculation of the Hardy function Z(t), using O(t<sup>0.25</sup>) standard operations. The repository contains scripts and source files for running the calculation using the [PARI library](https://pari.math.u-bordeaux.fr/).

## Prerequisites

1. A Linux environment with standard build tools (like `wget`, `make`, and `tar`) installed.
2. A Fortran compiler with quadmath capabilities and the MPI message passing library capabilities. 
   The scripts here assume you are using  `ftn`, the Cray Fortran compiler, using the GNU programming environment.
3. A SLURM workload manager for batch job submission.

### 1. Setup

To get started, clone the repository to your local machine and enter the directory:

```bash
git clone https://github.com/dml2391/Hardy-Z-t-code.git
cd Hardy-Z-t-code
```

### 2. Install the PARI Library

Run the `install_pari.sh` script. This script will:

- Download the PARI source code.
- Configure, compile, and install PARI in the current directory.
- Add necessary environment variables to your `.bashrc`.

```bash
bash install_pari.sh
```

### 3. Compile the Fortran Code

Once PARI is installed, you can compile the Fortran program using:

```bash
bash compile.sh
```


### 4. Run a quick test

Make sure it works (use ctrl+c to exit after the first iblock shows up):

```bash
bash run_test.sh
```


### 5. Submit a Job to SLURM

Before submitting, you need to ensure that the SLURM account in `zeta.pbs` matches your available SLURM account. To check your available account, run:

```bash
sacctmgr show assoc where user=$LOGNAME format=user,Account%12,MaxTRESMins,QOS%40
```

Then, modify the `#SBATCH --account=ChangeMe` line in `zeta.pbs` with your appropriate account details.

To submit the job, use:

```bash
sbatch zeta.pbs
```

File Descriptions:

* Fortran source files options are described in [src/README.md](src/README.mmd).
* `install_pari.sh`: Script to download, compile, and install PARI.
* `compile.sh`: Script to compile the Fortran program with PARI library linkage. **TO BE UPDATED**
* `zeta.pbs`: SLURM batch job script for running the compiled Fortran program. **TO BE UPDATED**

**Note**: The SLURM script will also check for the existence of a file named `zeta14v3resa` before executing the program. If the file does not exist, it will be created.

**Acknowledgments**

* Part of this work was performed with funding from the embedded CSE programme of the [ARCHER2 UK National Supercomputing Service](https://www.archer2.ac.uk/).
* This repo updates and supercedes the earlier repo: [HARDY](https://github.com/ashbre0/HARDY).
* [PARI/GP](https://pari.math.u-bordeaux.fr) - For providing the library used in this calculation.
