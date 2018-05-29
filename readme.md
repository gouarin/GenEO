[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/gouarin/GenEO/master)

This project solve the linear elasticity problem using PETSc in 2d and 3d for lamé coefficients which are constant or constant by cell on a cartesian grid.

# Installation

To install this package, you need first an installation of anaconda. If you don't have anaconda on your system, you can download miniconda for Python 3 (https://conda.io/miniconda.html).

Next, we will create an environment with all the needed packages using the following command.

    conda env create -f environment.yml

To activate your environment

    source activate petsc-elasticity

The final step is to install `petsc4py`. To do that, you have to specify the environment variable `PETSC_DIR`.

Run the following command in your terminal
    conda env list

    # conda environments:
    #
    ...
    petsc-elasticity      *  /home/loic/miniconda3/envs/petsc-elasticity
    ...

The path after the conda environment name `petsc-elasticity` is the path of `PETSC_DIR`.

    export PETSC_DIR=/home/loic/miniconda3/envs/petsc-elasticity

You can know install `petsc4py`

    pip install petsc4py

To install this project, you have to clone it

    git clone https://gitlab.centralesupelec.fr/gouarin/elasticity.git

Then

    cd elasticity
    python setup.py install

It's important to be in the conda environment created previously. If it is not the case

    source activate petsc-elasticity

# Execute demo file

In the directory of this project you have a `demos` directory with 2d and 3d examples.

This is an example of how to test one of them

    python elasticity_2d.py -ksp_monitor -pc_type gamg

- `ksp_monitor` indicates to PETSc to print the residual at each step.
- `pc_type gamg` indicates to PETSc to use an algebric multigrid as a preconditioner.

And in parallel

    mpiexec -n 4 python elasticity_2d.py -ksp_monitor -pc_type gamg

# Visualize the results

If the execution of `elasticity_2d.py` succeeded, you should have a file name `solution_2d.vts`. 

To visualize this file, you have to install paraview (https://www.paraview.org/download/).

- Start `paraview` and select file->load state. 
- Then select the file in the directory `paraview` of this project called `visu_2d.pvsm`.
- Then select the `vts` file.

You should see the results.
