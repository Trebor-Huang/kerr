import numpy as np
import matplotlib.pyplot as plt

MASS = 1.0
SPIN = 0.8
R_OUTER = MASS + np.sqrt(MASS*MASS - SPIN*SPIN)
R_INNER = MASS - np.sqrt(MASS*MASS - SPIN*SPIN)

ax = plt.figure().add_subplot(projection = "3d")

# The singularity ring at r = a
phi = np.linspace(0, 2*np.pi, num=100)
ax.plot(SPIN*np.cos(phi), SPIN*np.sin(phi), np.zeros_like(phi), lw=0.5)

# The horizons, we have (x^2 + y^2)/(r^2 + a^2) + z^2/r^2 = 1
for i, R in enumerate((R_OUTER, R_INNER)):
  theta, phi = np.meshgrid(np.linspace(0, np.pi/2), np.linspace(0, 2*np.pi))
  X = np.sin(theta) * np.cos(phi) * np.sqrt(R*R + SPIN*SPIN)
  Y = np.sin(theta) * np.sin(phi) * np.sqrt(R*R + SPIN*SPIN)
  Z = - np.cos(theta) * R
  ax.plot_surface(X, Y, Z, linewidth=0, color=(0.1 + 0.4*i,0.7-0.5*i,0.1,0.1+0.1*i)) # type: ignore

traj = np.loadtxt("./traj.out", delimiter=",")

DECIMATION = 10
I = traj[::DECIMATION,0]
T = traj[::DECIMATION,1]
X = traj[::DECIMATION,2]
Y = traj[::DECIMATION,3]
Z = traj[::DECIMATION,4]
R = traj[::DECIMATION,5]
ax.plot(X, Y, Z, lw=1)
maxt = np.max(np.abs(traj[:,2:5]))
ax.set(xlim=(-maxt, maxt), ylim=(-maxt, maxt), zlim=(-maxt, maxt))


ax2 = plt.figure().add_subplot()
ax2.plot(T, R)
ax2.plot(T, I/I[-1])

plt.show()
