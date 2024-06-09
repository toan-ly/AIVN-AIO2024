def is_unique(x, data):
    for i in data:
        if i == x:
            return 0
    return 1

def remove_duplicates(data):
    res = []
    for i in data:
        if is_unique(i, res):
            res.append(i)
    
    return res


lst = [10, 10, 9, 7, 7]
assert remove_duplicates(lst) == [10, 9, 7]

lst = [9, 9, 8, 1, 1]
print(remove_duplicates(lst))