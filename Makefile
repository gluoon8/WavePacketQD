# Makefile for wavepacket simulation

# Compiler 

help:
	@echo "make run - run the simulation"
	@echo "make prerequisites - install required packages"
	@echo "make clean - remove all output files"

run: 
	./python wavepack.py

prerequisites:
	pip install numpy
	pip install matplotlib
	pip install numba
	pip install imageio

clean:
	rm -f out/*.png
