# Makefile for wavepacket simulation

# Author: Manel Serrano

help:
	@echo "make run - run the simulation"
	@echo "make prerequisites - install required packages"
	@echo "make clean - remove all output files"

run: 
	python3 main.py

prerequisites:
	pip install numpy
	pip install matplotlib
	pip install numba
	pip install imageio

clean:
	rm -f out/WP*.png out/*FT.dat 
