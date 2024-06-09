from functions import *
import os
#import imageio
import imageio.v2 as imageio

# Set the working directory to the current folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Equation of motion of real part and imaginary part
integrate = 'euler'
dt = 1e-5
dx = 0.1
m = 1
h_bar = 1
L = 100
sigma_x = 5
w0 = 2
k0 = 3
x0 = 25
tfinal = 25

A = 1 / (2*np.pi*sigma_x**2)**0.25


Tsteps = int(tfinal/dt) 
Xsteps = int(L/dx) + 1
print('Number of Timesteps',Tsteps)               # Number of steps: 10000000
#Tsteps = 10000

welcome(integrate, dt, dx, L, A, x0, sigma_x, k0, w0, tfinal)

#----------------Inicialització dels paquets d'ones----------------

x, psiR, psiI, psiR_0, psiI_0 = initial_conditions(A, x0, sigma_x, k0, w0, L, dt, tfinal, dx)


psiR_aux = np.copy(psiR)
psiI_aux = np.copy(psiI)

norma = []
image_files = []


# Go to pics directory
os.chdir('data')


for i in range(0,Tsteps):


    if i % 10000 == 0:
        filename = plot_wave_packet(x, psiR, psiI, i, dt)
        image_files.append(filename)

    psiR, psiI, psiR_aux, psiI_aux = integrator(psiR, psiR_aux, psiI, psiI_aux, dt, dx, L, integrate)
    
    norma.append(norm(psiI, psiR))


print('len(norma)',len(norma))

    # Create the GIF using the image files
with imageio.get_writer('wavepacket.gif', mode='I') as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)


# Save the data to a file
time = np.linspace(0, Tsteps,Tsteps)
data = np.column_stack((time, norma))

# Save the data to a file
np.savetxt('output.txt', data, delimiter='\t', header='time\tnorma', comments='')

print('len(x)', len(x))

norma = np.round(norma)

plt.figure()
plt.plot(time, norma)
plt.title('Norm of the wave packet')
plt.savefig('norm.png')


plt.figure()
plt.plot(x, psiR,label='psi_Re')
plt.plot(x, psiI, label='psi_Im')
# plt.plot(x, psiR_0, label='psi_Im')
# plt.plot(x, psiI_0, label='psi_Im')
plt.legend()
plt.savefig('wavepacketfinal.png')