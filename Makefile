# Makefile for wavepacket simulation

# Compiler 

help:
	@echo "make run - run the simulation"
	@echo "make prerequisites - install required packages"
	@echo "make clean - remove all output files"

run: 
	python3 wavepack.py
	make clean

norm:
	python3 norm.py

prerequisites:
	pip install numpy
	pip install matplotlib
	pip install numba
	pip install imageio

clean:
	rm -f out/wave*.png
