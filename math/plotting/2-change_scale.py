#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.title("Exponential Decay of C-14")
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.semilogy(x, y, color='tab:blue')
ax.set_xlim([0, 28650])
plt.show()
