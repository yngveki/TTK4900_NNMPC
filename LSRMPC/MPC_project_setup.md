# Setting up python environments for MPC project
Note that this file is meant only for documentation purposes. I have not covered cases where several packages have the same package dependencies, so overlap may occur.

This guide is made during testing on Windows, and not yet tested on any Linux distribution, nor macOS.

### Setting up your virtual environment
Working in a virtual environment is not required, but neat in order to keep dependencies tidy, and avoid version issues and such.
I personally use miniconda for package and environment management, but using Python's built-in `venv` facilities could also work. Just note miniconda usually is safer, and may sometimes install packages in a way that actually works, where `pip` used in a Python `venv` does not.

Setting up your virtual environment with miniconda can be done as follows in any terminal (e.g. *git bash* or *VS Code*'s built-in terminal):

```
conda create -n your_env_name
```

Accessing your virtual environment can then be done by calling:

```
conda activate your_env_name
```

Now you are within your virtual environment, and further set-up (installation of packages) should be done from here.

Note that whenever conda might have issues, this might be due to too strict requirements for packages to be accepted into the conda libraries. In this case, simply replace the `conda` in commands with `pip`.

#### Installing miniconda
Install from their [page](https://docs.conda.io/en/latest/miniconda.html).

### Setting up MPC using the python control toolbox
Solving the MPC-problem, we formulate our own matrices, and feed these in a standard QP-manner into a QP that the package `qpsolvers` delivers. 

Follow the steps beneath to set up:

*Necessary*:
```
conda install numpy
conda install -c conda-forge qpsolvers
```

*Required for plotting*:
```
conda install matplotlib
```

### Setting up PyFMI
Based on the guide and documentation given [here](https://pypi.org/project/PyFMI/), this proved to be sufficient for me on Windows.

*Necessary*:
```
conda install numpy scipy lxml cython
conda install -c conda-forge assimulo
conda install -c conda-forge pyfmi
```

*Required for plotting functionalities*:
```
conda install matplotlib
```

### Compiling FMU-CS

```
from pymodelica import compile_fmu
fmu = compile_fmu('PalLib.Heidrun.ARTGLO.A23SEPTIC', {'C:\Modelica\PalLib.mo', 'C:\Modelica\StatoilLib2.mo'}, target ='cs', version='2.0', compiler_log_level='error', compiler_options={"variability_propagation":False})
```
mv: choke stengt på 16%, kan ikke gå høyere enn 100%. Kan ikke gå raskere enn ish 2% på 10s
mv: gaslift minimum 10% 
cv: minimum på 0, holde seg nære setpunkt