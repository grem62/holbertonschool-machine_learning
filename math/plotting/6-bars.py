#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

persons = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

fig, ax = plt.subplots()
bar_width = 0.5
bar_positions = np.arange(len(persons))

bottom = np.zeros(len(persons))

for i in range(fruit.shape[0]):
    ax.bar(bar_positions, fruit[i], bottom=bottom, color=colors[i],
           label=fruits[i], width=bar_width)
    bottom += fruit[i]

ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticks(bar_positions)
ax.set_xticklabels(persons)

ax.legend(["apples", "bananas", "oranges", "peaches"])

plt.show()
