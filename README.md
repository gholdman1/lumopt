# Continuous adjoint optimization wrapper for Lumerical

## Introduction

This is a continuous adjoint opimtization wrapper for Lumerical, using Python as the main user interface. It is released under an MIT license. It is still work in progress 
and any contribution will be very welcome! New features to come out soon, and make it even easier to use (hopefully)!

If you use this tool in any published work, please cite https://www.osapublishing.org/oe/abstract.cfm?uri=oe-21-18-21693 and give a link to this repo. Thanks!

## Tutorials, Examples, and Documentation

It is all here: https://lumopt.readthedocs.io/en/latest/

## Install

Make sure you have the latest version of Lumerical installed (it won't work correctly with older versions), and that lumapi (the python api) works

If you are using `conda`, first create a new environment `lumopt` and install `python`, `matplotlib`, and `jupyter`.

To install our branch of `lumopt`,

```bash
conda activate lumopt # optional
cd your/install/folder/
git clone -b brar-kats https://github.com/gholdman1/lumopt.git
cd lumopt
python setup.py develop
```

I would strongly recommend using jupyter notebooks to run optimizations.

## First optimization

If you are not using jupyter notebooks:

```bash
cd your/install/folder/examples/Ysplitter
python splitter_opt_2D.py
```

Otherwise copy `your/install/folder/examples/Ysplitter/splitter_opt_2D.py` into a notebook
