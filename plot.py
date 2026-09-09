import numpy as np
import matplotlib.pyplot as plt

traj = np.loadtxt("./traj.out", delimiter=",")
maxt = np.max(traj)
mint = np.min(traj)

ax = plt.figure().add_subplot(projection = "3d")
ax.plot(traj[::10,0], traj[::10,1], traj[::10,2], lw=2)

ax.set(xlim=(mint, maxt), ylim=(mint, maxt), zlim=(mint, maxt))
plt.show()
