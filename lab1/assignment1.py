import monkdata as m
import dtree as d

print("Assignment 1")
print(d.entropy(m.monk1))
print(d.entropy(m.monk2))
print(d.entropy(m.monk3))

print("Assignment 2")


print("Assignment 3")
print("MONK1")
for i in range(6):
    print(d.averageGain(m.monk1, m.attributes[i]))
print("MONK2")
for i in range(6):
    print(d.averageGain(m.monk2, m.attributes[i]))
print("MONK3")
for i in range(6):
    print(d.averageGain(m.monk3, m.attributes[i]))


# m.attributes[5]

# entropy of subset after splitting with attribute 5
print("Assignment 5")
for i in range(1, 5):
    subset5 = d.select(m.monk1, m.attributes[4], 2)  #nodes?
    print(d.entropy(subset5))
print(d.averageGain(subset5, m.attributes[0]))
