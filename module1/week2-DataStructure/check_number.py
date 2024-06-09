def check_number_presence(N):
    list_of_numbers = []
    results = ""
    for i in range(1, 5):
        list_of_numbers.append(i)

    if N in list_of_numbers:
        results = 'True'
    if N not in list_of_numbers:
        results = 'False'
    return results

# Check number occurrence in a list
def check_number_occurrence(integers, number=1):
    return any([True if num == number else False for num in integers])

def check_pos_helper(x):
    return 'T' if x > 0 else 'N'

def check_pos(data):
    return [check_pos_helper(x) for x in data]


if __name__ == '__main__':
    N = 7
    assert check_number_presence(N) == 'False'

    N = 2
    results = check_number_presence(N)
    print(results)

    my_list = [1, 3, 9, 4]
    assert check_number_occurrence(my_list, -1) == False

    my_list = [1, 2, 3, 4]
    print(check_number_occurrence(my_list, 2))

    data = [10, 0, -10, -1]
    assert check_pos(data) == ['T', 'N', 'N', 'N']

    data = [2, 3, 5, -1]
    print(check_pos(data))