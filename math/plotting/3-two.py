#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.title("Exponential Decay of Radioactive Elements")
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
ax.plot(x, y1, linestyle='dashed', color='red', label='C-14')
ax.plot(x, y2, color='green', label='Ra-226')
ax.set_xlim([0, 20000])
ax.set_ylim([0, 1.0])
plt.legend(loc="upper right")
plt.show()