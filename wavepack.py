from functions import *
import os
#import imageio


# Set the working directory to the current folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Equation of motion of real part and imaginary part
integrate = 'rk4'           # 'euler' or 'rk4'
potential = 'barrier'       # 'free' or 'barrier'
dt = 1e-4
dx = 0.1
m = 1
h_bar = 1
L = 100
sigma_x = 5
w0 = 2
k0 = 3
x0 = -20
tfinal = 40

A = 1 / (np.pi*sigma_x**2)**0.25


Tsteps = int(tfinal/dt) 
Xsteps = int(L/dx) + 1


welcome(integrate, dt, dx, L, A, x0, sigma_x, k0, w0, tfinal, potential)

#----------------Inicialització dels paquets d'ones----------------

x, psiR, psiI, psiR_0, psiI_0, V, pot = initial_conditions(A, x0, sigma_x, k0, w0, L, dt, tfinal, dx, potential)


psiR_aux = np.copy(psiR)
psiI_aux = np.copy(psiI)

norma = []
image_files = []


# Go to pics directory
os.chdir('out')
 

for i in range(0,Tsteps):


    if i % 10000 == 0:
        filename = plot_wave_packet(x, psiR, psiI, i, dt, pot, V)
        image_files.append(filename)
    #print('Time step: ', i)
    psiR, psiI, psiR_aux, psiI_aux = integrator(psiR, psiR_aux, psiI, psiI_aux, dt, dx, L, integrate, V)
    print('norm :', norm(psiI, psiR, dx))
    norma.append(norm(psiI, psiR, dx))


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
