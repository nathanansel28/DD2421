import dtree as d
import monkdata as m

monk1train, monk1val = d.partition(m.monk1, 0.6)

t = d.buildTree(monk1train, m.attributes)


# get initial score of tree
# prune tree
# get new score
# compare score
# if score is better then keep tree and prune again and repeat


initial_score = d.check(t, monk1val)
print("initial score: ", initial_score)
pruned_trees = d.allPruned(t)

for i in range(len(pruned_trees)):
    print(d.check(pruned_trees[i], monk1val))
