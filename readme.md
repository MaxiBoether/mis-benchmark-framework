# Independent Set Benchmarking Suite

This repository contains the code for our maximum independent set benchmarking suite as well as our implementations of the DGL-TreeSearch and the Gurobi-MWIS interface. In `solvers`, you can find the wrappers for the currently supported solvers (Gurobi, KaMIS, Intel-TreeSearch, DGL-Treesearch). In `data_generation`, you find the code required for generating random and real-world graphs.

For using this suite, `conda` is required. You can the `setup_bm_env.sh` script which will setup the conda environment with all required dependencies. You can find out more about the usage using `python main.py -h`. The `main.py` file is the main interface you will call for data generation, solving, and training.

In the `helper_scripts` folder, you find some scripts that could be helpful when doing analyses with this suite.