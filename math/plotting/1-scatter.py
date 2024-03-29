#!/usr/bin/env python3
""" script that makes a simple labeled scatter graph """
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# CREATE FIGURE
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.title("Men's Height vs Weight")
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
ax.scatter(x, y, color='magenta')
plt.show()
