def concat_lists(x, y):
    x.extend(y)
    return x


list_num1 = ['a', 2, 5]
list_num2 = [1, 1]
list_num3 = [0, 0]

assert concat_lists(list_num1, concat_lists(list_num2, list_num3)) == ['a', 2, 5, 1, 1, 0, 0]

list_num1 = [1, 2]
list_num2 = [3, 4]
list_num3 = [0, 0]

print(concat_lists(list_num1, concat_lists(list_num2, list_num3)))