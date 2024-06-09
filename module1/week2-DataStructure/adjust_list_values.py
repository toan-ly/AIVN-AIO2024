def adjust_list_values(data, max, min):
    res = []
    for i in data:
        if i < min:
            res.append(min)
        elif i > max:
            res.append(max)
        else:
            res.append(i)
    return res


if __name__ == '__main__':
    data = [5, 2, 5, 0, 1]
    assert adjust_list_values(data, max=1, min=0) == [1, 1, 1, 0, 1]

    data = [10, 2, 5, 0, 1]
    print(adjust_list_values(data, max=2, min=1))
            