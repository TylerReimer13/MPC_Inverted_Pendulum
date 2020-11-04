import numpy as np
import cvxpy as cp
from scipy import sparse
import matplotlib.pyplot as plt
from math import sin, cos
import matplotlib.animation as animation
from pendulum_dynamics import l, dt, sys_discrete, A_zoh, B_zoh


def plot_results():
    f = plt.figure()
    ax = f.add_subplot(211)
    plt.plot(cart_pos, label='cart pos')
    plt.plot(cart_vel, label='cart vel')
    plt.plot(pend_ang, label='pend ang')
    plt.plot(pend_ang_vel, label='pend ang vel')
    plt.ylabel(r"$(x_t)_1$", fontsize=16)
    plt.xticks([t_step for t_step in range(nsim) if t_step % 10 == 0])
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(ctrl_effort)
    plt.ylabel(r"$(u_t)_1$", fontsize=16)
    plt.xticks([t_step for t_step in range(nsim) if t_step % 10 == 0])
    plt.grid()

    plt.tight_layout()
    plt.show()


def animated_plot():
    fig = plt.figure()
    ax = plt.axes(xlim=(-5., 5.), ylim=(-.5, 2.))
    time_text = ax.text(2.75, 1.75, '')
    angle_text = ax.text(2.75, 1.65, '')
    pos_text = ax.text(2.75, 1.55, '')
    times = range(nsim)

    writer = animation.writers['ffmpeg']
    writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    def update(i):
        time_text.set_text("Time: {0:0}".format(round(times[i]*dt, 2)))
        angle_text.set_text("Angle: {0:0.2f}".format(pend_ang[i]*57.3))
        pos_text.set_text("Pos: {0:0.2f}".format(-cart_pos[i]))

        cart_x = -cart_pos[i]
        pend_pos = pend_ang[i]
        pendulum_x0 = cart_x
        pendulum_y0 = 0.
        pendulum_x1 = cart_x + l*sin(pend_pos)
        pendulum_y1 = l*cos(pend_pos)

        cart.set_data([cart_x-.15, cart_x+.15], 0.)
        pend.set_data([pendulum_x0, pendulum_x1], [pendulum_y0, pendulum_y1])
        ball.set_data(pendulum_x1, pendulum_y1)
        return (cart,) + (pend,) + (ball,)

    cart, = plt.plot([], [], 'black', linewidth=5.)
    pend, = plt.plot([], [], 'b-')
    ball, = plt.plot([], [], 'ro', markersize=8.)
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    line_ani = animation.FuncAnimation(fig, update, int(nsim), interval=100, repeat=True)
    plt.show()


def run_mpc():
    cost = 0.
    constr = [x[:, 0] == x_init]
    for t in range(N):
        cost += cp.quad_form(xr - x[:, t], Q) + cp.quad_form(u[:, t], R)
        constr += [cp.norm(u[:, t], 'inf') <= 10.]
        constr += [x[:, t + 1] == A_zoh * x[:, t] + B_zoh * u[:, t]]

    cost += cp.quad_form(x[:, N] - xr, Q)
    problem = cp.Problem(cp.Minimize(cost), constr)
    return problem


[nx, nu] = B_zoh.shape

Q = sparse.diags([10., 5., 100., 5.])
R = np.array([[.1]])

x0 = np.array([-1.5, -1., 0.65, 0.5])  # Initial conditions
xr = np.array([0., 0., 0., 0.])  # Desired states
xr *= -1

N = 10  # MPC Horizon length

x = cp.Variable((nx, N+1))
u = cp.Variable((nu, N))
x_init = cp.Parameter(nx)

nsim = 70  # Number of simulation timesteps
time = [0.]
cart_pos = [x0[0]]
cart_vel = [x0[1]]
pend_ang = [x0[2]]
pend_ang_vel = [x0[3]]
ctrl_effort = [u[:, 0].value]

for i in range(1, nsim+1):
    prob = run_mpc()
    x_init.value = x0
    print('TIME: ', round(i*dt, 2), 'STATES: ', [round(state, 2) for state in x0])
    prob.solve(solver=cp.OSQP, warm_start=True)
    x0 = A_zoh.dot(x0) + B_zoh.dot(u[:, 0].value)
    time.append(i)
    cart_pos.append(x0[0])
    cart_vel.append(x0[1])
    pend_ang.append(x0[2])
    pend_ang_vel.append(x0[3])
    ctrl_effort.append(u[:, 0].value)

plot_results()
animated_plot()
