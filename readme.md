# Independent Set Benchmarking Suite

This repository contains the code for our maximum independent set benchmarking suite as well as our implementations of the DGL-TreeSearch and the Gurobi-MWIS interface. In `solvers`, you can find the wrappers for the currently supported solvers (Gurobi, KaMIS, Intel-TreeSearch, DGL-Treesearch). In `data_generation`, you find the code required for generating random and real-world graphs.

For using this suite, `conda` is required. You can the `setup_bm_env.sh` script which will setup the conda environment with all required dependencies. You can find out more about the usage using `python main.py -h`. The `main.py` file is the main interface you will call for data generation, solving, and training.

In the `helper_scripts` folder, you find some scripts that could be helpful when doing analyses with this suite.

There are (of course) some improvements that can be made. For example, the argument parsing requires a major refactoring, and the output formats are currently not fully harmonized. We are open for pull requests, if you want to contribute. Thank you very much!

If you use this in your work, please cite us (and the papers of the solvers that you might use).

```bibtex
@inproceedings{boether_dltreesearch_2022,
  author = {Böther, Maximilian and Kißig, Otto and Taraz, Martin and Cohen, Sarel and Seidel, Karen and Friedrich, Tobias},
  title = {What{\textquoteright}s Wrong with Deep Learning in Tree Search for Combinatorial Optimization},
  booktitle = {Proceedings of the International Conference on Learning Representations ({ICLR})},
  year = {2022}
}
```
