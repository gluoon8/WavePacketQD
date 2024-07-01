from functions import *
import os
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))


#-------------------------System parameters------------------------
integrate = 'rk4'         # 'euler' or 'rk4'
potential = 'morse'      # 'free' or 'barrier' or 'harmonic' or 'morse'
dx = 0.1
m = 1
h_bar = 1
L = 100
sigma_x = 5
w0 = 2
k0 = 3
x0 = -20
tfinal = 18

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

norma_0 = norm(psiI, psiR, dx)
 
i = 0
write_mom('initialFT.dat',psiI, psiR, L)
plot_fourier('initialFT.dat', i, dt)

# Calculate time of calculation
start_time = time.time()

for i in range(0,Tsteps+1):


    if i % 10000 == 0:
        filename = plot_wave_packet(x, psiR, psiI, i, dt, pot, V, dx, L)
        image_files.append(filename)
        print('Process: ', i/Tsteps*100, '%')

    #print('Time step: ', i)
    psiR, psiI, psiR_aux, psiI_aux = integrator(psiR, psiR_aux, psiI, psiI_aux, dt, dx, L, integrate, V)
    #print('norm :', norm(psiI, psiR, dx))

    norma.append(norm(psiI, psiR, dx))

final_time = time.time() - start_time

filename = plot_wave_packet(x, psiR, psiI, i, dt, pot, V, dx, L)
image_files.append(filename)

    # Create the GIF using the image files
with imageio.get_writer('WP.gif', mode='I') as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)

# Write
write_mom('finalFT.dat',psiI, psiR, L)
plot_fourier('finalFT.dat', i, dt)

# Save the data to a file
time = np.linspace(0, Tsteps+1,Tsteps+1)
data = np.column_stack((time, norma))

# Calculate the coeficients of reflection and transmission
R, T, normafinal = coeficients_R_T(psiR, psiI, dx, Xsteps, pot, norma_0)

print('--------------END OF SIMULATION--------------')
print('')
print('Initial norm: ', norma_0)
print('Final norm: ', normafinal)
print('')
print('Reflection and transmission coefficients: ')
print('R = ', R)
print('T = ', T)

print('')
print('Time: ', final_time, 's')


#-----------TEST ZONE----------------

# Write psiR**2
# psi2 = np.column_stack((x, abs(psiR)**2 + abs(psiI)**2))
# np.savetxt('psi2.dat', psi2, delimiter='\t', header='x\tpsi2', comments='')


# Save the data of the norm to a file
np.savetxt('norm.dat', data, delimiter='\t', header='time\tnorma', comments='')
