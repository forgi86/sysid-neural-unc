# Neural state-space models: Empirical evaluation of uncertainty quantification

This repository contains the Python code to reproduce the results of the paper Neural state-space models: Empirical evaluation of uncertainty quantification
 by Marco Forgione, Dario Piga.


# Folders:
* [torchid](torchis):  PyTorch implementation of neural state-space model. Adapted from the library https://github.com/forgi86/pytorch-ident developed by the first author.
* [examples](examples): experimentations and analyses of the paper: Wiener-Hammerstein circuit and pick-and-place machine.
* [doepy](doepy): library used for experiment planning. Adapted (with small bug fixes) from https://github.com/tirthajyoti/doepy
 <!--*  [doc](doc): paper latex files -->

The [examples](examples) discussed in the paper are:

* [Wiener-Hammerstein Benchmark](examples/wh2009): A circuit with Wiener-Hammerstein behavior. Experimental dataset from http://www.nonlinearbenchmark.org

# Software requirements:
Experiments were performed on a Python 3.10 conda environment with

 * numpy
 * scipy
 * matplotlib
 * pandas
 * pytorch (version 1.12)
 
These dependencies may be installed through the commands:

```
conda install numpy scipy pandas matplotlib
conda install pytorch torchvision -c pytorch
```

To run the software, please make sure that this repository's root folder is added to 
your PYTHONPATH.
