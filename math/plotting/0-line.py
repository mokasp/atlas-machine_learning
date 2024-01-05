#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# CREATE FIGURE
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# PLOT Y
ax.plot(y, color='red')
# SET LIMIT TO X AXIS
ax.set_xlim([0, 10])
# DISPLAY GRAPH
plt.show()