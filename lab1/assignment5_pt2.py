import monkdata as m
import dtree as d


t1 = d.buildTree(m.monk1, m.attributes)
print(d.check(t1, m.monk1))
print(d.check(t1, m.monk1test))


t2 = d.buildTree(m.monk2, m.attributes)
print(d.check(t2, m.monk2))
print(d.check(t2, m.monk2test))


t3 = d.buildTree(m.monk3, m.attributes)
print(d.check(t3, m.monk3))
print(d.check(t3, m.monk3test))
