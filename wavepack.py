from functions import *
import os
#import imageio


# Set the working directory to the current folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Equation of motion of real part and imaginary part
integrate = 'rk4'           # 'euler' or 'rk4'
potential = 'free'       # 'free' or 'barrier' or 'harmonic' or 'morse'
dx = 0.1
m = 1
h_bar = 1
L = 100
sigma_x = 5
w0 = 2
k0 = 3
x0 = -20
tfinal = 24

#A = 1 / (np.pi*sigma_x**2)**0.25

#----------------Inicialització dels paquets d'ones----------------

x, psiR, psiI, psiR_0, psiI_0, V, pot, dt, Tsteps, Xsteps, sigma_x, A = initial_conditions(x0, sigma_x, k0, w0, L, tfinal, dx, potential, integrate)

welcome(integrate, dt, dx, L, A, x0, sigma_x, k0, w0, tfinal, potential)

psiR_aux = np.copy(psiR)
psiI_aux = np.copy(psiI)

norma = []
image_files = []

# Go to pics directory
os.chdir('out')
 
write_mom('initialFT.dat',psiI, psiR, L)


for i in range(0,Tsteps):


    if i % 10000 == 0:
        filename = plot_wave_packet(x, psiR, psiI, i, dt, pot, V, L, dx)
        image_files.append(filename)
        print('Process: ', i/Tsteps*100, '%')

    # Print the percentage of the process if the time step is a multiple of 1000



    #print('Time step: ', i)
    psiR, psiI, psiR_aux, psiI_aux = integrator(psiR, psiR_aux, psiI, psiI_aux, dt, dx, L, integrate, V)
    print('norm :', norm(psiI, psiR, dx))

    norma.append(norm(psiI, psiR, dx))


    # Create the GIF using the image files
with imageio.get_writer('wavepacket.gif', mode='I') as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)

cR, cI = fourier_transform(k0, psiR, psiI, L)

print('cR:', cR)
print('cI:', cI)
print('cR^2 + cI^2:', cR**2 + cI**2)

# Write
write_mom('finalFT.dat',psiI, psiR, L)

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
