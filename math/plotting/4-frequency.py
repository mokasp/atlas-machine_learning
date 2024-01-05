#!/usr/bin/env python3
""" script that plots a histogram/ frequencygraph """
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
params = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.title("Project A")
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.hist(student_grades, bins=params, edgecolor='black')
ax.set_xlim([0, 100])
ax.set_ylim([0, 30])
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.show()
