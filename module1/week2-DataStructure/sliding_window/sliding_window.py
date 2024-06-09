import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

# Brute Force O(nk)
def max_kernel(num_list: list[int], k: int) -> list[int]:
    res = []

    # Base condition
    n = len(num_list)
    if not num_list or k <= 0 or k > n:
        return res
    
    for i in range(n - k + 1):
        window = num_list[i:i + k]
        res.append(np.max(window))

    return res

# Optimized method with deque (first version)
def max_kernel_2(num_list: list[int], k: int) -> list[int]:
    res = []

    # Base condition
    n = len(num_list)
    if not num_list or k <= 0 or k > n:
        return res

    # Deque to store indices of elements in current window
    dq = deque() 
    
    for i in range(n):
        # Remove elements not in current window
        if dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove elements smaller than current element from back of the queue
        while dq and num_list[dq[-1]] < num_list[i]:
            dq.pop()

        # Add current element at back of the queue
        dq.append(i)

        # Front element in the queue is the maximum of current window
        if i >= k - 1:
            res.append(num_list[dq[0]])
        
    return res

# Optimized method with deque (second version)
def max_kernel_3(num_list: list[int], k: int) -> list[int]:
    res = []

    # Base condition
    n = len(num_list)
    if not num_list or k <= 0 or k > n:
        return res

    dq = deque()
    
    # Process initial window
    for i in range(k):
        while dq and num_list[dq[-1]] < num_list[i]:
            dq.pop()
        dq.append(i)
    res.append(num_list[dq[0]])

    # Process the rest of the list
    for i in range(k, n):
        if dq[0] <= i - k:
            dq.popleft()
        
        while dq and num_list[dq[-1]] < num_list[i]:
            dq.pop()

        dq.append(i)
        res.append(num_list[dq[0]])

    return res

def measure_time(func, num_list, k):
    start_time = time.time()
    func(num_list, k)
    end_time = time.time()
    return end_time - start_time
    

if __name__ == '__main__':
    assert max_kernel([3, 4, 5, 1, -44], 3) == [5, 5, 5]
    num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
    print(max_kernel(num_list, k=3))
        
    sizes = [10, 100, 500, 1000, 5000, 10000, 100000]
    k = 100

    times_1, times_2, times_3 = [], [], []
    for size in sizes:
        num_list = np.random.randint(-1000, 1000, size).tolist()

        time_1 = measure_time(max_kernel, num_list, k)
        time_2 = measure_time(max_kernel_2, num_list, k)
        time_3 = measure_time(max_kernel_3, num_list, k)

        times_1.append(time_1)
        times_2.append(time_2)
        times_3.append(time_3)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_1, label='Brute Force O(nk)')
    plt.plot(sizes, times_2, label='Optimized with Deque O(n) - Version 1')
    plt.plot(sizes, times_3, label='Optimized with Deque O(n) - Version 2')
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Complexity Comparison')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('complexity_comparison.png')
    plt.show()

    