Iterative LQR algorithm on a quadrotor

The goal of this project is to control a 2D quadrotor to perform acrobatic moves. It has been carried out in 4 parts of increasing complexity. The final part is the implementation of the iterative LQR (iLQR) algorithm.

Part-1

Computing u1* and  u2* such that the robot stays at a stationary position, where u1* and u2* are forces produced by the rotors.

Part-2

After the u* is calculated to keep the quad-rotor stationary, in this part I design a controller that ensures the stationary position of the quad-rotor even when pushed by random disturbance (e.g. due to wind). Therefore, designing a LQR controller to keep the quad-rotor at a predefined position.

Part-3

In this part my goal is to make the quad-rotor follow a desired trajectory.

Part-4

In this final part of the project my goal is to make the quad-rotor do a flip. This is achieved by using the iterative LQR algorithm.

<img src="https://github.com/arjune123/iLQR-QuadDynamics/blob/master/quadrotor.png" width="120">

where:

x is the horizontal and y being the vertical position of the quadrotor
θ is the orientation with respect to the horizontal plane
$v_x$ and $v_y$ are the linear velocities and $\\omega$ is the angular velocity of the robot
$u_1$ and $u_2$ are the forces produced by the rotors (our control inputs)
$m$ is the quadrotor mass
$I$ its moment of inertia (a scalar)
$r$ is the distance from the center of the robot frame to the propellers
$g$ is the gravity constant

We denote the state as $z = [x, v_x, y, v_y, \\theta, \\omega]^T$ 
and $u = [u_1, u_2]^T$.\n",
