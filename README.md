# MvCFN

This is an implementation of MvCFN from the paper "MvCFN: Multiview Contextual Feature Network with Complex Interaction for Knowledge Graph Completion".

## Requirements

Python is running at version 3.10. Other Python package versions can be found in `requirements.txt`.

It is recommended to create a virtual environment with the above version of Python using conda, and install the python packages in `requirements.txt` using pip in the virtual environment.

## Running a model

Parameters are configured in `config`, all the hyperparameters in the configuration file come from the paper.

Start training command:

```bash
# FB15k-237
python code/run.py

# WN18RR
python code/run.py dataset=WN18RR
