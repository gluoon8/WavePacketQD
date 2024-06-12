import os 
import numpy as np
import matplotlib.pyplot as plt

# set the directory as the current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('out')



data = np.loadtxt('./output.txt', skiprows=1)

plt.plot(data[:,0],data[:,1])
plt.savefig('norma.png')
plt.show()