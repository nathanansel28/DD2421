import monkdata as m
import dtree as d
from tabulate import tabulate

# Create rows for the table
rows = [
    ["MONK-1"] + [d.averageGain(m.monk1, m.attributes[i]) for i in range(6)],
    ["MONK-2"] + [d.averageGain(m.monk2, m.attributes[i]) for i in range(6)],
    ["MONK-3"] + [d.averageGain(m.monk3, m.attributes[i]) for i in range(6)],
]

# Headers
headers = ["Dataset", "a1", "a2", "a3", "a4", "a5", "a6"]

# Generate the table
table = tabulate(rows, headers=headers)
print(table)


# MONK-1 : a5
# MONK-2 : a5
# MONK-3 : a2


# ASSIGNMENT 4

'''
Entropy measures the level of uncertainty in a dataset—the higher the entropy, 
the greater the uncertainty. When an attribute that maximises information gain 
is selected, the weighted sum of the entropy of the resulting subsets is minimised, 
as shown in the equation. Since a reduction in entropy directly correlates with a 
reduction in uncertainty, the attribute with the highest information gain (the one 
that results in the lowest entropy) should be chosen for splitting. 

'''
