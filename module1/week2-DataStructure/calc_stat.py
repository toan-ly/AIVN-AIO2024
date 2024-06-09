import numpy as np
import time

def find_min(n):
    return np.min(n)

def find_min_2(n):
    return min(n)

def find_max(n):
    return np.max(n)

def find_avg(list_nums=[0, 1, 2]):
    var = 0
    for i in list_nums:
        var += i
    return var / len(list_nums)
    
def filter_divisible_by_3(data):
    var = []
    for i in data:
        if i % 3 == 0:
            var.append(i)
    return var

def measure_time(func, num_list):
    start_time = time.time()
    func(num_list)
    end_time = time.time()
    return end_time - start_time
  

if __name__ == '__main__':  
    # Find mininum
    my_list = [1, 22, 93, -100]
    assert find_min(my_list) == -100

    my_list = [1, 2, 3, -1]
    print(find_min(my_list))
    
    # Test with large array
    size = int(1e7)
    num_list = np.random.randint(-size//2, size//2, size=size)
    print('Time to find min in large array using min(): ', measure_time(find_min_2, num_list))
    print('Time to find min in large array using np.min(): ', measure_time(find_min, num_list))
    
    # Find maximum
    my_list = [1001, 9, 100, 0]
    assert find_max(my_list) == 1001

    my_list = [1, 9, 9, 0]
    print(find_max(my_list))

    # Find average
    assert find_avg([4, 6, 8]) == 6
    print(find_avg())

    # Filter element divisible by 3
    assert filter_divisible_by_3([3, 9, 4, 5]) == [3, 9]
    print(filter_divisible_by_3([1, 2, 3, 5, 6]))

    