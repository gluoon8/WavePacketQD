# Propagation of a 1D wave packet 

## Brief description of the project

This project involves the development of a code to simulate the propagation of a wave packet under a potential. 

Some examples are shown in the ***examples*** directory.

## Prerequisites
To execute the program, there are some pre-requisites, that can be installed on your first run on Linux with 'make prerequisites':
- Make: to execute the program (https://www.gnu.org/software/make/#download).
- Python 3.x : to carry out the simulation.
  - Numpy (https://numpy.org/install/)
  - Matplotlib (https://matplotlib.org/stable/users/installing/index.html)
  - Imageio (https://pypi.org/project/imageio/)
  - Numba (https://numba.pydata.org/)


## How to

1. Clone repository to your local host
2. Use `make` or `make help` to see available commands.
3. Before starting a simulation, change your parameters in main.py file  
5. To carry out the simulation, have a look to the ***quick quide***. 
6. Data is generated in out directory.


## Quick guide

To carry out a simulation after choosing parameters in wavepack.py file you can use:
```
make prerequisites 
make run
```
And results will appear in your /out directory!


## Help 
                          

- Commands:                                                       

  - `make run`: Runs the simulation.     
  
  - `make prerequisites`: Install required python modules to run the program:                              

  - `make clean`: Removes the output pictures        


Work developed in the Quantum Dynamics subject from [Master of Atomistic and Multiscale Computational Modelling in Physics, Chemistry and Biochemistry](http://www.ub.edu/computational_modelling/).

<table align="center">
  <tr>
    <td><img src="./examples/UB.png" alt="Logo UB"></td>
    <td><img src="./examples/UPC.png" alt="Logo UPC"></td>
  </tr>
  <tr>
    <td>Universitat de Barcelona</td>
    <td>Universitat Politècnica de Catalunya</td>
  </tr>
</table>