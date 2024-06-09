from sliding_window import max_kernel, max_kernel_2, max_kernel_3, measure_time
import time
import ast

# Test case from https://leetcode.com/problems/sliding-window-maximum/description/
file_path = 'sliding_window_test_case.txt'
with open(file_path, 'r') as f:
    nums = f.read().split(',')
    num_list = [int(num) for num in nums]
k = 50000

time_1 = measure_time(max_kernel, num_list, k)
time_2 = measure_time(max_kernel_2, num_list, k)
time_3 = measure_time(max_kernel_3, num_list, k)

print(f'Time for Brute Force: {time_1} seconds')
print(f'Time for Optimized with Deque - Version 1: {time_2} seconds')
print(f'Time for Optimized with Deque - Version 2: {time_3} seconds')

'''
Output:
Time for Brute Force: 123.03587794303894 seconds
Time for Optimized with Deque - Version 1: 0.037943124771118164 seconds
Time for Optimized with Deque - Version 2: 0.031287193298339844 seconds
'''
