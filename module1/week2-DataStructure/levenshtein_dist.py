import numpy as np
import time


# DP - Time: O(mn), Space: O(mn)
def levenshtein_distance_dp(token1: str, token2: str) -> int:
    m, n = len(token1) + 1, len(token2) + 1

    dist = np.zeros((m, n))

    for row in range(m):
        dist[row, 0] = row
    for col in range(n):
        dist[0, col] = col

    for row in range(1, m):
        for col in range(1, n):
            # No operations required as letters match
            if token1[row-1] == token2[col-1]:
                dist[row, col] = dist[row-1, col-1]
            else:
                dist[row, col] = 1 + min(dist[row-1, col], # deletion
                                           dist[row, col-1], # insertion
                                           dist[row-1, col-1]) # substitution
    
    return int(dist[-1, -1]) # bottom-right element in the dist

# BFS
def levenshtein_distance_bfs(token1: str, token2: str) -> int:
    m, n = len(token1), len(token2)

    to_visit = [(0, 0)] # Initial location
    visited = set()
    dist = 0

    while to_visit:
        next_lvl = []
        while to_visit:
            row, col = to_visit.pop()

            # Skip already visited locations
            if (row, col) in visited:
                continue

            # Move forward in both tokens as long as characters match
            while row < m and col < n and token1[row] == token2[col]:
                row += 1 
                col += 1
            
            # End of both tokens
            if row == m and col == n:
                return dist

            # Add possible locations to the next level
            if (row, col + 1) not in visited:
                next_lvl.append((row, col + 1)) # Insertion
            if (row + 1, col) not in visited:
                next_lvl.append((row + 1, col)) # Deletion
            if (row + 1, col + 1) not in visited:
                next_lvl.append((row + 1, col + 1)) # Substitution
                
            visited.add((row, col)) # Mark current location as visited

        dist += 1
        to_visit = next_lvl # Update queue with next level
            
def measure_time(func):
    start_time = time.time()
    assert func('hi', 'hello') == 4
    assert func('kitten', 'sitting') == 3
    assert func('horse', 'ros') == 3
    assert func('intention', 'execution') == 5
    assert func('zoologicoarchaeologist', 'zoogeologist') == 10
    assert func('abcdefghijklmnop', 'abcfghijklmnopqrs') == 5
    assert func('abcdef', 'uvwxyz') == 6
    assert func('abababababababab', 'babababababababa') == 2
    assert func('algorithms', 'smhtirogla') == 10
    end_time = time.time()
    return end_time - start_time
    

if __name__ == '__main__':
    assert levenshtein_distance_dp('hi', 'hello') == 4
    print(levenshtein_distance_dp('hola', 'hello'))
    
    print(f'Execution Time for Dynamic Programming (DP): {measure_time(levenshtein_distance_dp)} seconds')
    print(f'Execution Time for Breadth First Search (FPS): {measure_time(levenshtein_distance_bfs)} seconds')

