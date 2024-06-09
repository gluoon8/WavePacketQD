import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit

def real_psi(A, x0, sigma_x, k0, w0, x, t):
    return A*np.exp(-(x-x0)**2/(2*sigma_x**2))*np.cos(k0*x-w0*t)

def imaginary_psi(A, x0, sigma_x, k0, w0, x, t):
    return A*np.exp(-(x-x0)**2/(2*sigma_x**2))*np.sin(k0*x-w0*t)

def dens_prob(A, x0, sigma_x, x):
    return A**2*np.exp(-(x-x0)**2/(sigma_x**2))

def norm(psiI,psiR):
    return np.sum(abs(psiI)**2 + abs(psiR)**2) 

def initial_conditions(A, x0, sigma_x, k0, w0, L, dt, tfinal, dx):
    '''Initial conditions for the wave packet.
    
    - input:
        A: float, amplitude of the wave packet.
        x0: float, position of the wave packet.
        sigma_x: float, width of the wave packet.
        k0: float, wave number.
        w0: float, angular frequency.
        x: array, spatial grid.

    - output:
        x: array, spatial grid.
        psiR: array, real part of the wave packet at t = 0.
        psiI: array, imaginary part of the wave packet a t t = 0.
        psiR_0: array, real part of the wave packet at t = 0.
        psiI_0: array, imaginary part of the wave packet at t = 0.

    
    '''

    Xsteps = int(L/dx) + 1

    x = np.linspace(-L/2, L/2, Xsteps+1) 

    psiR = real_psi(A, x0, sigma_x, k0, w0, x, t=0)
    psiI = imaginary_psi(A, x0, sigma_x, k0, w0, x, t=0)
    
    psiR[0] = 0.       # Psi is 0 at the borders.
    psiR[-1] = 0.       
    psiI[0] = 0.
    psiI[-1] = 0.

    psiR_0 = np.copy(psiR)
    psiI_0 = np.copy(psiI)

    # Plot initial conditions:
    plt.figure()
    plt.title('t = 0 s')
    plt.plot(x, psiR_0,label='psi_Re')
    plt.plot(x, psiI_0, label='psi_Im')
    plt.xlabel('t')
    plt.ylabel('psi')
    plt.legend()
    plt.savefig('wavepacketinitial.png')

    return x, psiR, psiI, psiR_0, psiI_0
    
def welcome(integrate, dt, dx, L, A, x0, sigma_x, k0, w0, tfinal):
    print('----------------------')
    print('')
    print('WAVE PACKET SIMULATION')
    print('')
    print('----------------------')
    print('Author: Manel Serrano Rodríguez')
    print('Subject: Quantum Dynamics')
    print('')
    print('----------PARAMETERS----------')
    print('')
    print(f'integration method: {integrate}')
    print(f'dt:     {dt} s')
    print(f'dx:     {dx} m')
    print(f'L:      {L} m')
    print(f'x0:     {x0} m')
    print(f'sigma:  {sigma_x} m')
    print(f'k0:     {k0} m')
    print(f'w0:     {w0} m')
    print(f'')    
    print(f'simulation time: {tfinal} s')
    print(f'')
    print('-----------------------------')



@njit
def integrator(psiR, psiR_aux, psiI, psiI_aux, dt, dx, L, integrate):
    h_bar = 1
    m = 1
    Xsteps = int(L/dx)    
    factor = dt*h_bar/(2*m * dx**2)

    psiR_aux = np.copy(psiR)
    psiI_aux = np.copy(psiI)

    if integrate == 'euler':
        for j in range(1,Xsteps-1):

            psiR[j] = psiR_aux[j] - factor*(psiI_aux[j+1]-2*psiI_aux[j]+psiI_aux[j-1]) 
            psiI[j] = psiI_aux[j] + factor*(psiR_aux[j+1]-2*psiR_aux[j]+psiR_aux[j-1]) 

    return psiR, psiI, psiR_aux, psiI_aux



def plot_wave_packet(x, psiR, psiI, i, dt):
    '''
    Plot the wave packet at time i*dt.

    '''

    plt.figure()
    plt.plot(x, psiR, label='psi_Re')
    plt.plot(x, psiI, label='psi_Im')
    plt.legend()
    plt.title(f'Time: {i*dt:2f} s')
    plt.ylim(-0.3, 0.3)
    filename = f'wavepacket{i}.png'
    plt.savefig(filename)
    plt.close()

    return filename

