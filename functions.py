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

def get_potential(potential, x, Xsteps, x0, sigma):
    if potential == 'free':
        V = np.zeros(Xsteps+1)
        pot = 0

    elif potential == 'barrier':
        V = np.zeros(Xsteps+1)
        V[int(Xsteps/2):int(Xsteps/2)+50] = 5.5  # Barrier in the middle of the grid.

        pot = 1

    elif potential == 'harmonic':
        m=1
        n = 15
        w = 4.505 / (n + 1/2)
        V = 0.5*m*w**2*(x-x0)**2
        pot = 2

    elif potential ==  'morse':
        sigma = 3
        D = 10
        a = 0.075
        V = D * (1 - np.exp(-a*(x-x0)))**2
        pot = 3


    else:
        raise ValueError('Potential not defined')

    return V, pot, sigma

def initial_conditions(x0, sigma_x, k0, w0, L, tfinal, dx, potential, integrate):
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
        dt = 0.5e-5
    elif integrate == 'rk4':
        dt = 1e-4

    A = 1 / (np.pi*sigma_x**2)**0.25
    Xsteps = int(L/dx) + 1
    Tsteps = int(tfinal/dt)
    #print('Xsteps', Xsteps)
    x = np.linspace(-L/2, L/2, Xsteps+1) 

    psiR = real_psi(A, x0, sigma_x, k0, w0, x, t=0)
    psiI = imaginary_psi(A, x0, sigma_x, k0, w0, x, t=0)
    
    psiR[0] = 0.       # Psi is 0 at the borders.
    psiR[-1] = 0.       
    psiI[0] = 0.
    psiI[-1] = 0.

    psiR_0 = np.copy(psiR)
    psiI_0 = np.copy(psiI)

    V, pot, sigma_x = get_potential(potential, x, Xsteps, x0, sigma_x)



    return x, psiR, psiI, psiR_0, psiI_0, V, pot, dt, Tsteps, Xsteps, sigma_x, A
    
def welcome(integrate, dt, dx, L, A, x0, sigma_x, k0, w0, tfinal, potential):
    print('-----------------------------------')
    print('')
    print('       WAVE PACKET SIMULATION')
    print('')
    print('-----------------------------------')
    print('Author: Manel Serrano Rodríguez')
    print('Subject: Quantum Dynamics')
    print('Universitat de Barcelona')
    print('-----------------------------------')
    print('')   
    print('www.github.com/gluoon8/WavePacketQD')
    print('')
    print('-------------PARAMETERS------------')
    print('')
    print(f'integration method: {integrate}')
    print(f'potential: {potential}')
    print(f'dt:     {dt}')
    print(f'dx:     {dx}')
    print(f'L:      {L}')
    print(f'x0:     {x0}')
    print(f'sigma:  {sigma_x}')
    print(f'k0:     {k0}')
    print(f'w0:     {w0}')
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

    plt.figure(figsize=(5, 2.5), dpi=100)
    plt.ylabel('$\psi$')
    

    if pot == 1:
        plt.plot(x, V*0.1, label='Barrier')
        plt.plot(x, psiR , label='$\psi(x)_R$', color='blue', linewidth=1)
        plt.plot(x, psiI , label='$\psi(x)_I$', color='red' , linewidth=1)
        plt.plot(x, abs(psiR)**2 + abs(psiI)**2 , label='$|\psi(x)|^2$', color='green', linewidth=2)
        plt.ylim(-0.5, 0.5)
        plt.xlabel('x')

    elif pot == 2:
        plt.plot(x, V, label='Harmonic potential')
        plt.plot(x, 3*psiR + 4.5, label='$\psi(x)_R$', color='blue', linewidth=1)
        plt.plot(x, 3*psiI + 4.5, label='$\psi(x)_I$', color='red', linewidth=1)
        plt.plot(x, abs(psiR)**2 + abs(psiI)**2+ 4.5, label='$|\psi(x)|^2$',color='green', linewidth=2)
        plt.ylim(-0.5, 7)
        plt.xlabel('x')
    elif pot == 3:
        Vplot = np.copy(V)
        plt.plot(x, Vplot * 0.8, label='Morse potential')
        plt.plot(x, 3*psiR +4.5 , label='$\psi(x)_R$', color='blue', linewidth=1)
        plt.plot(x, 3*psiI + 4.5, label='$\psi(x)_I$', color='red', linewidth=1)
        plt.plot(x, 8*(abs(psiR)**2 + abs(psiI)**2)+ 4.5, label='$|\psi(x)|^2$',color='green', linewidth=2)
        plt.ylim(-0.5, 10)
        plt.xlabel('x')
    else:
        plt.plot(x, psiR, label='$\psi(x)_R$', color='blue', linewidth=1)
        plt.plot(x, psiI, label='$\psi(x)_I$', color='red', linewidth=1)
        plt.plot(x, abs(psiR)**2 + abs(psiI)**2, label='$|\psi(x)|^2$', color='green', linewidth=2)
        plt.ylim(-0.4, 0.4)
        plt.xlabel('x')


    T, R = transmission_coefficient(psiR, psiI, dx, L)
    #plt.text(0, 0.4, f'T = {T:.2f}, R = {R:.2f}', fontsize=12)
    #plt.legend(loc='upper right',shadow=True, fancybox=True)
    if i == 0:
        plt.legend(loc='upper right',shadow=True, fancybox=True)

    plt.xlim(-L/2, L/2)
    plt.title(f't = {int(i*dt)}')
    plt.xlabel('x')
    plt.tight_layout()
    filename = f'WP_{i*dt}.png'
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


def fourier_transform(k_in, psiR, psiI, L):
    '''Calculate the Fourier coefficients for a given wave number k_in.
    
    - input:
        k_in: float, input wave number.
        psiR: numpy array, real part of the wave packet.
        psiI: numpy array, imaginary part of the wave packet.
        L: float, length of the grid.
    
    - output:
        cR: float, real part of the Fourier coefficient.
        cI: float, imaginary part of the Fourier coefficient.
    '''
    k = -k_in
    Nx = psiR.size
    dx = L / (Nx - 1)

    cR = 0.5 * dx * (psiR[0] + psiR[-1] * np.cos(k * L) - psiI[-1] * np.sin(k * L))
    cI = 0.5 * dx * (psiI[0] + psiR[-1] * np.sin(k * L) + psiI[-1] * np.cos(k * L))
    for i in range(1, Nx - 1):
        x = dx * i
        cR += dx * (psiR[i] * np.cos(k * x) - psiI[i] * np.sin(k * x))
        cI += dx * (psiR[i] * np.sin(k * x) + psiI[i] * np.cos(k * x))
    cR /= (2 * np.pi)
    cI /= (2 * np.pi)

    return cR, cI


def write_mom(file,psiI, psiR, L):
    Nk = 1000
    k0 = 3
    kb = 1.5*k0
    ka = -kb

    dk = (kb - ka) / (Nk -1)
    with open(file, 'w') as f:
        f.write('#k, cR, cI\n')
        for i in range(Nk):
            k = ka + i*dk
            cR, cI = fourier_transform(k, psiR, psiI, L)
            f.write(f'{k}    {cR}    {cI}    {cR**2 + cI**2}\n')
        f.close()
    print(f'Fourier transform written to file {file}')    

def plot_fourier(file, i, dt):
    data = np.loadtxt(file,skiprows=1)
    plt.figure(figsize=(5, 2.5), dpi=100)
    plt.grid(alpha=0.5)
    plt.plot(data[:,0], data[:,3], color='#FC0A57', linewidth=1.5)
    plt.xlabel('k')
    plt.ylabel('A(k)')
    plt.ylim(-0.025, 0.6)
    plt.xlim(-4.5, 4.5)         # Change the limits of the x-axis 
    plt.tight_layout()
    plt.savefig(f'TF_{i*dt}.png')
    plt.close()


def coeficients_R_T(psiR, psiI, dx, Xsteps, pot, norma_0):
    if pot == 1: 
        normafinal = norm(psiI[:int(Xsteps/2)], psiR[:int(Xsteps/2)], dx)
    
    if pot == 2:
        normafinal = norm(psiI[180:420], psiR[180:420], dx)
    
    if pot == 3:
        # normafinal = norm(psiI[300:511], psiR[300:511], dx)   # Per a t = 11
        normafinal = norm(psiI[220:470], psiR[220:470], dx)


    if pot == 0:
        normafinal = norm(psiI, psiR, dx)

    R = normafinal/norma_0
    T = 1 - R
    return R, T, normafinal