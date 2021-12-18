# Ising model in 2D
This code is written in Python3 to use monte carlo simulation for the crystal growth of Ising model.

## Prerequisites
A virtual python environment is recommended
Follow the guideline for anaconda users.
```
conda create -ny ising python==3.8
conda activate ising
pip3 install tqdm matplotlib imageio
```
* imageio: image combining for generating GIF
* matplotlib: math plotting library
* tqdm: library for progress bar (optional)

## How to run
This code requires you to move to the src/ working directory for better file organisations.
### 1. Local updating algorithm
The input of variables are in command line.
```
cd src && python3 MCS.py --N 50 --beta 0.2
```
Running this command produces the record_local/ folder that saves the GIF files under different values of beta value folders.
### 2. Block updating algorithm
```
cd src && python3 init.py --N 50 --beta 0.2
```
Running this command produces the record/ folder that saves the GIF files under different values of beta value folders.