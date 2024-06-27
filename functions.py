import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
import imageio.v2 as imageio

def real_psi(A, x0, sigma_x, k0, w0, x, t):
    return A*np.exp(-(x-x0)**2/(2*sigma_x**2))*np.cos(k0*x-w0*t)

def imaginary_psi(A, x0, sigma_x, k0, w0, x, t):
    return A*np.exp(-(x-x0)**2/(2*sigma_x**2))*np.sin(k0*x-w0*t)

def dens_prob(A, x0, sigma_x, x):
    return A**2*np.exp(-(x-x0)**2/(sigma_x**2))

def norm(psiI,psiR, dx):
    return np.sum(abs(psiI)**2 + abs(psiR)**2) * dx 

def get_potential(potential, x, Xsteps, x0):
    if potential == 'free':
        V = np.zeros(Xsteps+1)
        pot = 0

    elif potential == 'barrier':
        V = np.zeros(Xsteps+1)
        V[int(Xsteps/2):int(Xsteps/2)+50] = 4  # Barrier in the middle of the grid.

        pot = 1

    elif potential == 'harmonic':
        m=1
        n = 15
        w = 4.505 / (n + 1/2)
        V = 0.5*m*w**2*(x-x0)**2
        pot = 2

    elif potential ==  'morse':
        D = 10
        a = 0.075
        V = D * (1 - np.exp(-a*(x-x0)))**2
        pot = 3


    else:
        raise ValueError('Potential not defined')

    return V, pot

def initial_conditions(A, x0, sigma_x, k0, w0, L, tfinal, dx, potential, integrate):
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
        psiI: array, imaginary part of the wave packet at t = 0.
        psiR_0: array, real part of the wave packet at t = 0.
        psiI_0: array, imaginary part of the wave packet at t = 0.

    
    '''
    if integrate == 'euler':
        dt = 1e-5
    elif integrate == 'rk4':
        dt = 1e-4


    Xsteps = int(L/dx) + 1
    Tsteps = int(tfinal/dt)

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
    plt.close()

    V, pot = get_potential(potential, x, Xsteps, x0)



    return x, psiR, psiI, psiR_0, psiI_0, V, pot, dt, Tsteps, Xsteps
    
def welcome(integrate, dt, dx, L, A, x0, sigma_x, k0, w0, tfinal, potential):
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
    print(f'potential: {potential}')
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
def compute_k(psiR, psiI, psiR_aux, psiI_aux, dt, dx, L, V):
    h_bar = 1
    m = 1
    Xsteps = int(L/dx)
    factor = h_bar/(2*m * dx**2)


    k1r = np.zeros(len(psiR))
    k1i = np.zeros(len(psiI))

    # print('k1r', k1r.shape)
    # print('k1i', k1i.shape)
# 
    # print('psiI_aux', psiI_aux.shape)
    # print('psiR_aux', psiR_aux.shape)
# 
    # print('psiI', psiI.shape)
    # print('psiR', psiR.shape)
    # print('V', V.shape)



    for j in range(1,Xsteps-1):

        k1r[j] = -  factor * (psiI_aux[j+1] - 2*psiI_aux[j] + psiI_aux[j-1]) + V[j]*psiI_aux[j] / h_bar
        k1i[j] =    factor * (psiR_aux[j+1] - 2*psiR_aux[j] + psiR_aux[j-1]) - V[j]*psiR_aux[j] / h_bar

    psiI_aux = np.copy(psiI + 0.5*dt*k1i)
    psiR_aux = np.copy(psiR + 0.5*dt*k1r)

    k2r = np.zeros(len(psiR), dtype=np.float64)
    k2i = np.zeros(len(psiI), dtype=np.float64)

    for j in range(1,Xsteps-1):

        k2r[j] = -  factor * (psiI_aux[j+1] - 2*psiI_aux[j] + psiI_aux[j-1]) + V[j]*psiI_aux[j] / h_bar
        k2i[j] =    factor * (psiR_aux[j+1] - 2*psiR_aux[j] + psiR_aux[j-1]) - V[j]*psiR_aux[j] / h_bar


    psi2i = psiI + 0.5*dt*k2i
    psi2r = psiR + 0.5*dt*k2r

    psiI_aux = np.copy(psi2i)
    psiR_aux = np.copy(psi2r)

    k3r = np.zeros(len(psiR), dtype=np.float64)
    k3i = np.zeros(len(psiI), dtype=np.float64)

    for j in range(1,Xsteps-1):
        
        k3r[j] = -  factor * (psiI_aux[j+1] - 2*psiI_aux[j] + psiI_aux[j-1]) + V[j]*psiI_aux[j] / h_bar
        k3i[j] =    factor * (psiR_aux[j+1] - 2*psiR_aux[j] + psiR_aux[j-1]) - V[j]*psiR_aux[j] / h_bar

    psiI_aux = np.copy(psiI + dt*k3i)
    psiR_aux = np.copy(psiR + dt*k3r)

    k4r = np.zeros(len(psiR), dtype=np.float64)
    k4i = np.zeros(len(psiI), dtype=np.float64)

    for j in range(1,Xsteps-1):
        
        k4r[j] = -  factor * (psiI_aux[j+1] - 2*psiI_aux[j] + psiI_aux[j-1]) + V[j]*psiI_aux[j] / h_bar
        k4i[j] =    factor * (psiR_aux[j+1] - 2*psiR_aux[j] + psiR_aux[j-1]) - V[j]*psiR_aux[j] / h_bar
    

    return k1r, k2r, k3r, k4r, k1i, k2i, k3i, k4i


@njit
def integrator(psiR, psiR_aux, psiI, psiI_aux, dt, dx, L, integrate, V):
    h_bar = 1
    m = 1
    Xsteps = int(L/dx)    
    factor = dt*h_bar/(2*m * dx**2)

    psiR_aux = np.copy(psiR)
    psiI_aux = np.copy(psiI)

    if integrate == 'euler':
        for j in range(1,Xsteps-1):

            psiR[j] = psiR_aux[j] - factor*(psiI_aux[j+1]-2*psiI_aux[j]+psiI_aux[j-1]) + V[j]*psiI_aux[j]*dt
            psiI[j] = psiI_aux[j] + factor*(psiR_aux[j+1]-2*psiR_aux[j]+psiR_aux[j-1]) - V[j]*psiR_aux[j]*dt

    elif integrate == 'rk4':
        k1r, k2r, k3r, k4r, k1i, k2i, k3i, k4i = compute_k(psiR, psiI, psiR_aux, psiI_aux, dt, dx, L, V)
        psiR = psiR + dt/6. * (k1r + 2*k2r + 2*k3r + k4r)
        psiI = psiI + dt/6. * (k1i + 2*k2i + 2*k3i + k4i)
       

    return psiR, psiI, psiR_aux, psiI_aux






def plot_wave_packet(x, psiR, psiI, i, dt,pot, V, dx, L):
    '''
    Plot the wave packet at time i*dt.

    '''

    plt.figure()

    if pot == 1:
        plt.plot(x, V, label='Barrier')
        plt.plot(x, psiR + 4.5, label='psi_Re')
        plt.plot(x, psiI + 4.5, label='psi_Im')
        plt.plot(x, abs(psiR)**2 + abs(psiI)**2 + 4.5, label='|psi|^2')
        plt.ylim(-0.5, 7)

    elif pot == 2:
        plt.plot(x, V, label='Harmonic potential')
        plt.plot(x, psiR + 4.5, label='psi_Re')
        plt.plot(x, psiI + 4.5, label='psi_Im')
        plt.plot(x, abs(psiR)**2 + abs(psiI)**2+ 4.5, label='|psi|^2')
        plt.ylim(-0.5, 7)
    elif pot == 3:
        Vplot = np.copy(V)
        plt.plot(x, Vplot, label='Morse potential')
        plt.plot(x, psiR + 4.5, label='psi_Re')
        plt.plot(x, psiI + 4.5, label='psi_Im')
        plt.plot(x, abs(psiR)**2 + abs(psiI)**2+ 4.5, label='|psi|^2')
        plt.ylim(-0.5, 10)
    else:
        plt.plot(x, psiR, label='psi_Re')
        plt.plot(x, psiI, label='psi_Im')
        plt.plot(x, abs(psiR)**2 + abs(psiI)**2, label='|psi|^2')
    
    T, R = transmission_coefficient(psiR, psiI, dx, L)
    plt.text(0, 0.4, f'T = {T:.2f}, R = {R:.2f}', fontsize=12)
    plt.legend()
    plt.xlim(-L/2 * 1000, L/2 * 1000)
    plt.title(f'Time: {i*dt:2f}')
    
    filename = f'wavepacket{i}.png'
    plt.savefig(filename)
    plt.close()

    return filename

# Calculate the transmission coefficient and reflection coefficient for the barrier potential.

def transmission_coefficient(psiR, psiI, dx, L):
    '''Calculate the transmission coefficient for the barrier potential.
    
    - input:
        psiR: array, real part of the wave packet.
        psiI: array, imaginary part of the wave packet.
        dx: float, spatial step.
        L: float, length of the grid.
    
    - output:
        T: float, transmission coefficient.
        R: float, reflection coefficient.
    
    '''
    Xsteps = int(L/dx)
    T = np.sum(abs(psiI[Xsteps//2+50])**2 + abs(psiR[Xsteps//2+50])**2) * dx
    R = np.sum(abs(psiI[Xsteps//2-50])**2 + abs(psiR[Xsteps//2-50])**2) * dx

    return T, R


def fourier_transform(psiR, psiI, dx, L):
    '''Calculate the Fourier transform of the wave packet.
    
    - input:
        psiR: array, real part of the wave packet.
        psiI: array, imaginary part of the wave packet.
        dx: float, spatial step.
        L: float, length of the grid.
    
    - output:
        k: array, wave number grid.
        psiR_k: array, real part of the wave packet in the wave number space.
        psiI_k: array, imaginary part of the wave packet in the wave number space.
    
    '''
    Xsteps = int(L/dx)
    k = np.fft.fftfreq(Xsteps+2, d=dx)
    k = np.fft.fftshift(k)
    psiR_k = np.fft.fft(psiR)
    psiI_k = np.fft.fft(psiI)

    return k, psiR_k, psiI_k

def plot_fourier_transform(k, psiR_k, psiI_k):
    '''Plot the Fourier transform of the wave packet.
    
    - input:
        k: array, wave number grid.
        psiR_k: array, real part of the wave packet in the wave number space.
        psiI_k: array, imaginary part of the wave packet in the wave number space.
    
    '''
    plt.figure()
    plt.plot(k, psiR_k, label='psi_Re(k)')
    plt.plot(k, psiI_k, label='psi_Im(k)')
    plt.legend()
    plt.title('Fourier transform of the wave packet')
    plt.savefig('fouriertransform.png')
    plt.close()

