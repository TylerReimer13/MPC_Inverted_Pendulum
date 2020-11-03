import numpy as np
import control

"""
Dynamic equations taken from 'https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlDigital'
"""

M = 0.5  # Cart mass
m = 0.2  # Pendulum mass
b = 0.1  # Coefficient of friction for cart
I = 0.006  # Mass moment of inertia of the pendulum
g = 9.8  # Gravity
l = 0.3  # Length to pendulum center of mass
dt = .1  # Time step

p = I*(M+m)+M*m*l**2

A = np.array([[0,      1,              0,            0],
              [0, -(I+m*l**2)*b/p,  (m**2*g*l**2)/p, 0],
              [0,      0,              0,            1],
              [0, -(m*l*b)/p,       m*g*l*(M+m)/p,   0]])

B = np.array([[0],
              [(I+m*l**2)/p],
              [0],
              [m*l/p]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

D = np.array([[0],
              [0]])

sys = control.StateSpace(A, B, C, D)
sys_discrete = control.c2d(sys, dt, method='zoh')

A_zoh = np.array(sys_discrete.A)
B_zoh = np.array(sys_discrete.B)
