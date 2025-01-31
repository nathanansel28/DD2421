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
