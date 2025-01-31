import monkdata as m
import dtree as d
from tabulate import tabulate

# based on information gain values from assignment 3, attribute 5 should be picked = m.attributes[4]
# value table
rows = []
for i in range(1, 5):  # Attribute values are 1, 2, 3, 4
    entropy_value = d.entropy(d.select(m.monk1, m.attributes[4], i))
    rows.append([i, entropy_value])  # Append attribute value and entropy

# Headers for the table
headers = ['Attribute Value', 'Entropy']

# Generate the table
# Use grid format for clarity
table = tabulate(rows, headers=headers)
# print(table)


# based on results from above, attribute 5 with value 2 provides the greatest entropy
subsets = []

for i in range(1, 5):
    subsets.append(d.select(m.monk1, m.attributes[4], i))

rows = [
    ["a5=1"] + [d.averageGain(subsets[0], m.attributes[i]) for i in range(6)],
    ["a5=2"] + [d.averageGain(subsets[1], m.attributes[i]) for i in range(6)],
    ["a5=3"] + [d.averageGain(subsets[2], m.attributes[i]) for i in range(6)],
    ["a5=4"] + [d.averageGain(subsets[3], m.attributes[i]) for i in range(6)],
]

# Headers
headers = ["Attribute Value", "a1", "a2", "a3", "a4", "a5", "a6"]

# Generate the table
table = tabulate(rows, headers=headers)
print(table)
