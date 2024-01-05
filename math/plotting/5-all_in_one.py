#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure(constrained_layout=True)

gs = fig.add_gridspec(nrows=3, ncols=2)

fig.suptitle('All in One')

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(y0, color='red')
ax1.set_xlim([0, 10])


ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlabel('Height (in)', fontsize=8)
ax2.set_ylabel('Weight (lbs)', fontsize=8)
ax2.set_title("Men's Height vs Weight", fontsize=8)
ax2.scatter(x1, y1, color='magenta')

ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlabel('Time (years)', fontsize=8)
ax3.set_ylabel('Fraction Remaining', fontsize=8)
ax3.set_title('Exponential Decay of C-14', fontsize=8)
ax3.semilogy(x2, y2)
ax3.set_xlim([0, 28650])

ax4= fig.add_subplot(gs[1, 1])
ax4.set_xlabel('Time (years)', fontsize=8)
ax4.set_ylabel('Fraction Remaining', fontsize=8)
ax4.set_title('Exponential Decay of Radioactive Elements', fontsize=8)
ax4.plot(x3, y31, linestyle='dashed', color='red', label='C-14')
ax4.plot(x3, y32, color='green', label='Ra-226')
ax4.set_xlim([0, 20000])
ax4.set_ylim([0, 1])
ax4.legend(loc="upper right", fontsize=8)

ax5 = fig.add_subplot(gs[2, :])
ax5.set_xlabel('Grades', fontsize=8)
ax5.set_ylabel('Number of Students', fontsize=8)
ax5.set_title('Project A', fontsize=8)
ax5.hist(student_grades, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], edgecolor='black')
ax5.set_xlim([0, 100])
ax5.set_ylim([0, 30])

plt.show()
