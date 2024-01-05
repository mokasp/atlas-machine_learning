#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

color = ['red', 'yellow', '#ff8000', '#ffe5b4']
x = ['Farrah', 'Fred', 'Felicia']
label = ['apples', 'bananas', 'oranges', 'peaches']

fig, ax = plt.subplots()

for i in range(fruit.shape[0]):
    ax.bar(x, fruit[i], color=color[i],
           label=label[i], width=0.5, bottom=np.sum(fruit[:i], axis=0))
ax.set_ylim([0, 80])
plt.legend(loc="upper right")
plt.title('Number of Fruit per Person')
plt.show()
