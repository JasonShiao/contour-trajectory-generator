import matplotlib.pyplot as plt
import numpy as np
from utilities import *
from curve_generator import parametric_curve_generator

from matplotlib.animation import FuncAnimation

def animate_trajectory(xd, yd, t, title=None):
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    line, = ax.plot([], [], lw=2, label="Parametric Curve")
    point, = ax.plot([], [], 'ro', label="Current Point")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)
    ax.legend()
    ax.set_title('Parametric Curve Animation')

    # Update function for the animation
    def update(frame):
        # Update the line and point
        line.set_data(xd[:frame], yd[:frame])
        #point.set_data(xd[-1], yd[-1])
        return line, point

    # Create the animation
    num_frames = len(t)
    ani = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=True)
    # Show the animation
    plt.show()


def main():

    ### Step 1: Trajectory Generation
    x_t, y_t, dx_t, dy_t = parametric_curve_generator("rabbit.jpg", 0.3, debug=True)

    # T, xd, yd of lissajous curve, this define the trajectory (not related to the actual velocity)
    T = 2*np.pi
    t = np.linspace(0, T, 3000)
    xd = x_t(t)
    yd = y_t(t)
    
    animate_trajectory(xd, yd, t)
    
    #xd_bridge = np.array([])
    #yd_bridge = np.array([])
    #for i in range(1, len(xd)):
    #    xd_bridge = np.append(xd_bridge, ((xd[i][0] - xd[i-1][-1])/T*t + xd[i-1][-1]) )
    #    yd_bridge = np.append(yd_bridge, ((yd[i][0] - yd[i-1][-1])/T*t + yd[i-1][-1]) )

    # Calculate the arc length: Sum of short segments
    d = 0
    for i in range(1, len(t)):
        d += np.sqrt((xd[i] - xd[i-1])**2 + (yd[i] - yd[i-1])**2)
    
    print(f"Curve length: {d}")
    # Define the desired speed by (curve length / desired time period)
    # calculate average velocity
    #c = d/tfinal
    c = 0.14
    tfinal = np.array(d)/c
    
    # Assert that the average velocity is less than 0.25
    print(f'Average velocity: {c}')
    assert(c < 0.25)
    
    ta = 0.4 # or some constant?
    
    # forward euler to calculate alpha
    dt = 0.002
    error_compensation = 0.04
    t = np.arange(0, tfinal+error_compensation, dt)
    alpha = np.zeros(len(t))
    for i in range(1, len(t)):
        xdot = dx_t(alpha[i-1])
        ydot = dy_t(alpha[i-1])
        alpha_dot = c*trapezoid(t[i], tfinal+error_compensation, ta)/np.sqrt(xdot**2 + ydot**2)
        alpha[i] = alpha[i-1] + alpha_dot*dt # (equation 7)
    
    # Avoid numerical error at the end
    alpha[-1] = T
    
    # plot alpha vs t
    plt.plot(t, alpha,'b-',label='alpha')
    plt.plot(t, np.ones(len(t))*T, 'k--',label='T (period)')
    plt.xlabel('t')
    plt.ylabel('alpha')
    plt.title('alpha vs t')
    plt.legend()
    plt.grid()
    plt.show()
        
    # rescale our trajectory with alpha
    x = x_t(alpha)
    y = y_t(alpha)
    
    print(f"x_0: {x_t(0)}, y_0: {y_t(0)}")
    print(f"x_end: {x_t(alpha[-1])}, y_end: {y_t(alpha[-1])}")
    print(f"x_2pi: {x_t(2*np.pi)}, y_2pi: {y_t(2*np.pi)}")

    # calculate velocity
    xdot = np.diff(x)/dt
    ydot = np.diff(y)/dt
    v = np.sqrt(xdot**2 + ydot**2)

    # plot velocity vs t
    plt.plot(t[1:], v, 'b-',label='velocity')
    plt.plot(t[1:], np.ones(len(t[1:]))*c, 'k--',label='average velocity')
    plt.plot(t[1:], np.ones(len(t[1:]))*0.25, 'r--',label='velocity limit')
    plt.xlabel('t')
    plt.ylabel('velocity')
    plt.title('velocity vs t')
    plt.legend()
    plt.grid()
    plt.show()
        
    ### Step 2: Forward Kinematics
    L1 = 0.2435
    L2 = 0.2132
    W1 = 0.1311
    W2 = 0.0921
    H1 = 0.1519
    H2 = 0.0854

    M = np.array([[-1, 0, 0, L1 + L2],
                  [0, 0, 1, W1 + W2],
                  [0, 1, 0, H1 - H2],
                  [0, 0, 0, 1]])
    
    S1 = np.array([0, 0, 1, 0, 0, 0])
    S2 = np.array([0, 1, 0, -H1, 0, 0])
    S3 = np.array([0, 1, 0, -H1, 0, L1])
    S4 = np.array([0, 1, 0, -H1, 0, L1 + L2])
    S5 = np.array([0, 0, -1, -W1, L1+L2, 0])
    S6 = np.array([0, 1, 0, H2-H1, 0, L1+L2])
    S = np.array([S1, S2, S3, S4, S5, S6]).T
    
    B1 = np.linalg.inv(ECE569_Adjoint(M))@S1
    B2 = np.linalg.inv(ECE569_Adjoint(M))@S2
    B3 = np.linalg.inv(ECE569_Adjoint(M))@S3
    B4 = np.linalg.inv(ECE569_Adjoint(M))@S4
    B5 = np.linalg.inv(ECE569_Adjoint(M))@S5
    B6 = np.linalg.inv(ECE569_Adjoint(M))@S6
    B = np.array([B1, B2, B3, B4, B5, B6]).T

    theta0 = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])
    
    # perform forward kinematics using ECE569_FKinSpace and ECE569_FKinBody
    T0_space = ECE569_FKinSpace(M, S, theta0)
    print(f'T0_space: {T0_space}')
    T0_body = ECE569_FKinBody(M, B, theta0)
    print(f'T0_body: {T0_body}')
    T0_diff = T0_space - T0_body
    print(f'T0_diff: {T0_diff}')
    T0 = T0_body
    
    # calculate Tsd for each time step
    Tsd = np.zeros((4, 4, len(t)))
    for i in range(len(t)):
        Tsd[:, :, i] = np.eye(4)
        Tsd[0, 3, i] = x[i]
        Tsd[1, 3, i] = -y[i]
        Tsd[:, :, i] = T0@Tsd[:, :, i]
    
    # plot p(t) vs t in the {s} frame
    xs = Tsd[0, 3, :]
    ys = Tsd[1, 3, :]
    zs = Tsd[2, 3, :]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, 'b-',label='p(t)')
    ax.plot(xs[0], ys[0], zs[0], 'go',label='start')
    ax.plot(xs[-1], ys[-1], zs[-1], 'rx',label='end')
    ax.set_title('Trajectory in s frame')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend()
    plt.show()


    ### Step 3: Inverse Kinematics

    # when i=0
    thetaAll = np.zeros((6, len(t)))

    initialguess = theta0
    eomg = 1e-6
    ev = 1e-6

    # From home position to the first point
    thetaSol, success = ECE569_IKinBody(B, M, Tsd[:,:,0], initialguess, eomg, ev)
    if not success:
        raise Exception(f'Failed to find a solution at index {0}')
    thetaAll[:, 0] = thetaSol

    # when i=1...,N-1
    for i in range(1, len(t)):
        # Use previous solution as current guess
        initialguess = thetaSol

        thetaSol, success = ECE569_IKinBody(B, M, Tsd[:,:,i], initialguess, eomg, ev)
        if not success:
            raise Exception(f'Failed to find a solution at index {i}')
        thetaAll[:, i] = thetaSol

    # verify that the joint angles don't change much
    dj = np.diff(thetaAll, axis=1)
    plt.plot(t[1:], dj[0], 'b-',label='joint 1')
    plt.plot(t[1:], dj[1], 'g-',label='joint 2')
    plt.plot(t[1:], dj[2], 'r-',label='joint 3')
    plt.plot(t[1:], dj[3], 'c-',label='joint 4')
    plt.plot(t[1:], dj[4], 'm-',label='joint 5')
    plt.plot(t[1:], dj[5], 'y-',label='joint 6')
    plt.xlabel('t (seconds)')
    plt.ylabel('first order difference')
    plt.title('Joint angles first order difference')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # verify that the joint angles will trace out our trajectory
    actual_Tsd = np.zeros((4, 4, len(t)))
    for i in range(len(t)):
        # use forward kinematics to calculate Tsd from our thetaAll
        actual_Tsd[:,:,i] = ECE569_FKinBody(M, B, thetaAll[:, i])
    
    xs = actual_Tsd[0, 3, :]
    ys = actual_Tsd[1, 3, :]
    zs = actual_Tsd[2, 3, :]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, 'b-',label='p(t)')
    ax.plot(xs[0], ys[0], zs[0], 'go',label='start')
    ax.plot(xs[-1], ys[-1], zs[-1], 'rx',label='end')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Verified Trajectory in s frame')
    ax.legend()
    plt.show()

    # save to csv file (you can modify the led column to control the led)
    # led = 1 means the led is on, led = 0 means the led is off
    led = np.ones_like(t)
    data = np.column_stack((t, thetaAll.T, led))
    np.savetxt('jshiao_bonus.csv', data, delimiter=',')


if __name__ == "__main__":
    main()
