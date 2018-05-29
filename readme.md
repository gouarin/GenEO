[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/gouarin/GenEO/master)

This project solve the linear elasticity problem using PETSc in 2d and 3d for lamé coefficients which are constant or constant by cell on a cartesian grid.

# Installation

To install this package, you need first an installation of anaconda. If you don't have anaconda on your system, you can download miniconda for Python 3 (https://conda.io/miniconda.html).

To install this project, you have to clone it

    git clone https://github.com/gouarin/GenEO.git
    cd GenEO

Next, we will create an environment with all the needed packages using the following command.

    conda env create -f environment.yml

To activate your environment

    source activate petsc-geneo

Then

    python setup.py install

# Execute demo file

In the directory of this project you have a `demos` directory with 2d and 3d examples.

It's important to be in the conda environment created previously. If it is not the case

    source activate petsc-geneo

This is an example of how to test one of them

    mpiexec -np 4 python demo_elasticity_2d.py -AMPCG_verbose -ksp_monitor -PCBNN_verbose

# Visualize the results

If the execution of `demo_elasticity_2d.py` succeeded, you should have a file name `'solution_2d_asm.vts'`. 

To visualize this file, you have to install paraview (https://www.paraview.org/download/).

- Start `paraview` and select file->load state. 
- Then select the file in the directory `paraview` of this project called `visu_2d.pvsm`.
- Then select the `vts` file.

You should see the results.
