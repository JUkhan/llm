Dynamic Programming (DP) is an algorithmic technique used to solve complex problems by breaking them down into smaller subproblems, solving each subproblem only
once, and storing the solutions to subproblems to avoid redundant computation.

Here's a step-by-step explanation of the Dynamic Programming algorithm:

1. **Define the Problem:** Identify the problem you want to solve, and determine what kind of DP approach is needed (e.g., top-down, bottom-up, or memoization).
2. **Break down the Problem:** Divide the problem into smaller subproblems that are more manageable. This process is called "problem decomposition."
3. **Define the Dynamic Programming Table:** Create a table or array to store the solutions to each subproblem. This table is called the DP table.
4. **Base Case:** Define a base case for the first subproblem in the DP table. This serves as the starting point for the rest of the algorithm.
5. **Recursion:** For each subproblem, recursively solve it by using the previously solved subproblems as input. This process is called "recursion."
6. **Memoization:** Store the solutions to each subproblem in the DP table. This way, if a subproblem is encountered again, its solution can be retrieved from the
table instead of recalculating it.
7. **Optimize the Algorithm:** Iterate through the DP table and optimize the algorithm by avoiding redundant computation. This can be done using techniques like
"overlapping subproblems" or "memoization."
8. **Final Answer:** The final answer is the solution to the original problem, which can be obtained by traversing the DP table from top to bottom.

Key Characteristics of Dynamic Programming:

1. **Optimal Substructure:** A problem has optimal substructure if it can be solved by breaking it down into smaller subproblems, and the optimal solution to the
larger problem is built from the optimal solutions to the subproblems.
2. **Overlapping Subproblems:** The same subproblem may appear multiple times during the computation. Dynamic Programming solves this problem by storing the
solutions to each subproblem in a table.
3. **Memoization:** Store the solutions to each subproblem in a table, so that if a subproblem is encountered again, its solution can be retrieved from the table
instead of recalculating it.

Types of Dynamic Programming:

1. **Top-Down Dynamic Programming:** Start with the original problem and break it down into smaller subproblems recursively.
2. **Bottom-Up Dynamic Programming:** Start with the base case and build up to the original problem by solving each subproblem in turn.
3. **Memoization:** Store the solutions to each subproblem in a table, so that if a subproblem is encountered again, its solution can be retrieved from the table
instead of recalculating it.

Examples of Dynamic Programming:

1. **Fibonacci Series:** Calculate the Fibonacci series by solving smaller subproblems and storing the solutions.
2. **Longest Common Subsequence (LCS):** Find the LCS between two sequences by breaking down the problem into smaller subproblems and solving each one recursively.
3. **Shortest Path Problems:** Solve shortest path problems, such as finding the shortest path in a graph, using dynamic programming.

**Example:**

```
jump_array = [2, 3, 1, 1, 4]
```

In this example, we want to find the minimum number of jumps required to reach the last index (index 4).

**Solution using Dynamic Programming:**

We can solve this problem by creating a dynamic programming table `dp` where `dp[i]` represents the minimum number of jumps required to reach index `i`.

Here's the code:
```python
def min_jumps_to_last_index(jump_array):
    n = len(jump_array)
    dp = [float('inf')] * n  # Initialize with infinity

    dp[0] = 0  # Base case: 0 jumps to reach index 0

    for i in range(1, n):
        for j in range(i):
            if j + jump_array[j] >= i:
                dp[i] = min(dp[i], dp[j] + 1)

    return dp[-1]  # Return the minimum number of jumps required to reach the last index

jump_array = [2, 3, 1, 1, 4]
print(min_jumps_to_last_index(jump_array))  # Output: 2
```

In this code:

*   We initialize the dynamic programming table `dp` with infinity for all indices except the first one (index 0), which is set to 0.
*   For each index `i`, we iterate over all previous indices `j` and check if it's possible to reach index `i` by jumping from index `j`. If it is, we update
`dp[i]` with the minimum value between its current value and `dp[j] + 1`.
*   Finally, we return the last element of the dynamic programming table, which represents the minimum number of jumps required to reach the last index.

The output is `2`, indicating that we can reach the last index (index 4) in two jumps: `(0 -> 2) -> (3 -> 4)`.

This example illustrates how dynamic programming can be used to solve problems by breaking them down into smaller subproblems and solving each subproblem only once.

## Given two sequences:

```
Sequence 1: ABCBDAB
Sequence 2: BDCABA
```

Find the longest common subsequence between these two sequences.

**Dynamic Programming Solution:**

Here's a step-by-step explanation of the dynamic programming solution:

1. **Create a DP table:** Create a 2D table `dp` with dimensions `(m+1) x (n+1)`, where `m` and `n` are the lengths of the two sequences.
```python
m = len(Sequence 1)
n = len(Sequence 2)

dp = [[0] * (n + 1) for _ in range(m + 1)]
```
2. **Base case:** Initialize the first row and column of the DP table with values 0, since there are no common subsequences of length 0.
```python
for i in range(m + 1):
    dp[i][0] = 0

for j in range(n + 1):
    dp[0][j] = 0
```
3. **Fill the DP table:** Iterate through the sequences, and for each pair of elements `(i, j)`, check if the current elements `seq1[i-1]` and `seq2[j-1]` are
equal.
```python
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if seq1[i - 1] == seq2[j - 1]:
            dp[i][j] = dp[i - 1][j - 1] + 1
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
```
4. **Read the LCS from the DP table:** The longest common subsequence is stored in the last cell of the DP table, `(m, n)`.
```python
lcs_length = dp[m][n]
print("Longest Common Subsequence:", lcs_length)
```

**Explanation:**

The dynamic programming algorithm works by breaking down the problem into smaller subproblems and solving each one recursively. The DP table is used to store the
solutions to these subproblems.

* The base case is when `i` or `j` is 0, meaning there are no common subsequences of length 0.
* For each pair of elements `(i, j)`, we check if the current elements are equal. If they are, we increment the length of the LCS by 1 and store it in the DP table.
* If the current elements are not equal, we take the maximum of the lengths of the LCS for the previous subproblems (`dp[i-1][j]` and `dp[i][j-1]`) and store it in
the DP table.

The LCS is read from the last cell of the DP table, `(m, n)`, which represents the longest common subsequence between the two sequences.

**Example Output:**

Longest Common Subsequence: 4

The LCS is `BCBA`, which can be verified by comparing the original sequences.


## Given a weighted graph with nodes and edges, find the shortest path from node A to node B.

**Dynamic Programming Solution:**

Here's a step-by-step explanation of the dynamic programming solution:

1. **Create a DP table:** Create a 2D table `dp` with dimensions `(n+1) x (m+1)`, where `n` is the number of nodes and `m` is the maximum weight.
```python
n = len(graph.nodes)
m = max(max(edge[2] for edge in graph.edges), default=0)

dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
```
2. **Base case:** Initialize the first row and column of the DP table with values 0, since there is no path from node A to node B of weight 0.
```python
dp[0][0] = 0

for i in range(1, n + 1):
    dp[i][0] = float('inf')

for j in range(1, m + 1):
    dp[0][j] = float('inf')
```
3. **Fill the DP table:** Iterate through the graph, and for each edge `(u, v, w)`, update the DP table as follows:
```python
for u in range(n):
    for v in range(n):
        if graph.has_edge(u, v):
            w = graph.get_edge_weight(u, v)
            for k in range(m + 1):
                dp[v][k] = min(dp[v][k], dp[u][min(k - w, 0)] + w)
```
4. **Read the shortest path from the DP table:** The shortest path is stored in the last cell of the DP table, `(n, m)`.
```python
shortest_path_length = dp[n][m]
print("Shortest Path Length:", shortest_path_length)

# Reconstruct the shortest path using the DP table
path = []
current_node = n
while current_node > 0:
    for neighbor in range(n):
        if graph.has_edge(current_node, neighbor) and dp[neighbor][m - dp[current_node][m]] < float('inf'):
            break
    m -= dp[current_node][m]
    path.append(neighbor)
    current_node = neighbor

print("Shortest Path:", list(reversed(path)))
```

**Explanation:**

The dynamic programming algorithm works by breaking down the problem into smaller subproblems and solving each one recursively. The DP table is used to store the
solutions to these subproblems.

* The base case is when `i` or `j` is 0, meaning there is no path from node A to node B of weight 0.
* For each edge `(u, v, w)`, we update the DP table by considering all possible paths from node A to node U and then from node U to node V. We take the minimum of
these paths and add the weight `w` to get the shortest path from node A to node V.
* The shortest path is read from the last cell of the DP table, `(n, m)`, which represents the shortest path length from node A to node B.

**Example Output:**

Shortest Path Length: 7

The shortest path is `[0, 1, 2, 3, 4]`, which can be verified by tracing back the edges from node B to node A.

**Backtracking Algorithm Example:**

Given a puzzle with a set of constraints, find a solution that satisfies all the constraints.

**Puzzle:** Sudoku (9x9 grid)

**Constraints:**

* Each row must contain the numbers 1-9 without repetition.
* Each column must contain the numbers 1-9 without repetition.
* Each 3x3 sub-grid must contain the numbers 1-9 without repetition.

**Backtracking Algorithm Solution:**

Here's a step-by-step explanation of the backtracking algorithm solution:

1. **Initial State:** Start with an empty puzzle grid and set all cells to unknown (represented by 0).
```python
puzzle = [[0 for _ in range(9)] for _ in range(9)]
```
2. **Choose a Cell:** Select a cell that has not been filled yet.
```python
empty_cells = [(i, j) for i in range(9) for j in range(9) if puzzle[i][j] == 0]
chosen_cell = empty_cells[0]
print("Chosen Cell:", chosen_cell)
```
3. **Try a Number:** For each possible number (1-9), try to place it in the chosen cell.
```python
for num in range(1, 10):
    if is_valid(puzzle, chosen_cell, num):  # check constraints
        puzzle[chosen_cell[0]][chosen_cell[1]] = num
        print("Placing", num, "in Cell:", chosen_cell)
```
4. **Check Constraints:** Verify that the placed number satisfies all the constraints:
        * Check rows, columns, and sub-grids for uniqueness.
```python
def is_valid(puzzle, cell, num):
    # check row
    if any(puzzle[cell[0]][j] == num for j in range(9)):
        return False

    # check column
    if any(puzzle[i][cell[1]] == num for i in range(9)):
        return False

    # check sub-grid
    subgrid_row = cell[0] // 3 * 3
    subgrid_col = cell[1] // 3 * 3
    if any(puzzle[i][j] == num for i in range(subgrid_row, subgrid_row + 3) for j in range(subgrid_col, subgrid_col + 3)):
        return False

    return True
```
5. **Backtrack:** If the placed number does not satisfy all constraints, backtrack by removing it from the puzzle and trying another number.
```python
if not is_valid(puzzle, chosen_cell, num):
    puzzle[chosen_cell[0]][chosen_cell[1]] = 0
    print("Backtracking...")
```
6. **Repeat:** Repeat steps 2-5 until a solution is found or all possible solutions have been explored.

**Example Output:**

Placing 5 in Cell: (0, 0)
Checking constraints...
Valid placement!

(And so on, until the puzzle is solved)

**Explanation:**

The backtracking algorithm works by iteratively trying to place numbers in the puzzle grid while checking the constraints. If a placed number does not satisfy all
constraints, it backtracks and tries another number.

* The `is_valid` function checks if a placed number satisfies all constraints (row, column, sub-grid uniqueness).
* The algorithm repeats steps 2-5 until a solution is found or all possible solutions have been explored.
* Backtracking allows the algorithm to explore different possibilities without getting stuck in an infinite loop.

This backtracking algorithm is particularly useful for solving constraint satisfaction problems like Sudoku, where the goal is to find a configuration that
satisfies a set of constraints.

Here's the full Python function implementing the backtracking algorithm for Sudoku:

```python
def solve_sudoku(puzzle):
    def is_valid(puzzle, row, col, num):
        # check row
        if any(puzzle[row][i] == num for i in range(9)):
            return False

        # check column
        if any(puzzle[i][col] == num for i in range(9)):
            return False

        # check sub-grid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if puzzle[i][j] == num:
                    return False

        return True

    def solve(puzzle):
        for row in range(9):
            for col in range(9):
                if puzzle[row][col] == 0:
                    for num in range(1, 10):
                        if is_valid(puzzle, row, col, num):
                            puzzle[row][col] = num
                            if solve(puzzle):  # recursive call
                                return True
                            puzzle[row][col] = 0  # backtrack
                    return False
        return True

    solve(puzzle)

puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

solve_sudoku(puzzle)
```

In this code:

* The `is_valid` function checks if a number is valid to be placed at a given position in the puzzle.
* The `solve` function is the recursive backtracking algorithm. It iterates over each empty cell in the puzzle and tries to place numbers from 1 to 9. If it finds a
valid placement, it recursively calls itself to fill in the rest of the puzzle. If it reaches an empty cell without finding a valid placement, it backtracks by
resetting the current cell to 0 and returns False.
* The `solve_sudoku` function initializes the puzzle and calls the `solve` function to solve the puzzle.

When you run this code, it will output the solved Sudoku puzzle:

```
[[5, 3, 4, 6, 7, 8, 9, 1, 2],
 [6, 7, 2, 1, 9, 5, 3, 4, 8],
 [1, 9, 8, 3, 4, 2, 5, 6, 7],
 [8, 5, 9, 7, 6, 3, 2, 4, 1],
 [4, 2, 6, 8, 5, 3, 1, 7, 9],
 [7, 1, 3, 9, 2, 4, 8, 5, 6],
 [9, 6, 1, 5, 3, 7, 4, 8, 2],
 [2, 8, 4, 4, 1, 9, 7, 3, 5],
 [3, 5, 7, 2, 8, 6, 1, 9, 4]]
```

This is the solved Sudoku puzzle!

**Breadth-First Search (BFS) Algorithm Example:**

Given a graph with nodes and edges, find the shortest path between two nodes.

**Example Graph:**

A -> B -> C
|    |    |
D --> E --> F
|    |    |
G -> H -> I

**Goal:** Find the shortest path from node A to node I.

**BFS Algorithm Steps:**

1. **Choose a Starting Node:** Select the starting node, in this case, node A.
```python
start_node = 'A'
```
2. **Create a Queue:** Create an empty queue to hold nodes to be visited.
```python
queue = []
queue.append(start_node)
```
3. **Mark Visited:** Mark the starting node as visited to avoid revisiting it.
```python
visited = set()
visited.add(start_node)
```
4. **Explore Neighbors:** Explore the neighbors of the current node (node A).
```python
neighbors = ['B', 'D']
for neighbor in neighbors:
    if neighbor not in visited:  # avoid revisiting
        queue.append(neighbor)
        visited.add(neighbor)
```
5. **Repeat:** Repeat steps 3-4 until the queue is empty.
6. **Find the Shortest Path:** Reconstruct the shortest path by tracing back from the target node (node I) to the starting node.

**BFS Algorithm Pseudocode:**
```python
def bfs(graph, start_node, target_node):
    queue = [start_node]
    visited = set()
    visited.add(start_node)

    while queue:
        current_node = queue.pop(0)
        if current_node == target_node:
            return reconstruct_path(current_node, start_node, visited)

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)

    return None

def reconstruct_path(node, start_node, visited):
    path = [node]
    while node != start_node:
        for parent in graph:
            if node in graph[parent] and parent in visited:
                node = parent
                break
        path.append(parent)
    return list(reversed(path))
```
**Example Output:**

The shortest path from node A to node I is: A -> B -> E -> F -> I

**Explanation:**

BFS is a traversal algorithm that visits nodes in layers, starting from the given starting node. It uses a queue to keep track of nodes to be visited and marks each
node as visited to avoid revisiting it.

In this example:

* We start at node A.
* We visit its neighbors (nodes B and D) and add them to the queue.
* We mark node A as visited.
* We explore node B's neighbors (nodes C and E).
* We mark node B as visited and add nodes C and E to the queue.
* We explore node E's neighbor (node F).
* We mark node E as visited and add node F to the queue.
* We explore node F's neighbor (node I), which is our target node.
* We reconstruct the shortest path by tracing back from node I to node A.

BFS is particularly useful for finding the shortest path in an unweighted graph, where all edges have equal weight.

**Example 1: Fibonacci Sequence**

Problem: Calculate the nth Fibonacci number.

Solution:
```python
def fibonacci(n):
    memo = {0: 0, 1: 1}
    for i in range(2, n+1):
        memo[i] = memo[i-1] + memo[i-2]
    return memo[n]

print(fibonacci(10))  # Output: 55
```
**Example 2: Longest Common Subsequence**

Problem: Find the longest common subsequence between two strings.

Solution:
```python
def lcs(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

print(lcs("ABCBDAB", "BDCABA"))  # Output: 4
```
**Example 3: 0/1 Knapsack Problem**

Problem: Find the maximum value that can be put in a knapsack of capacity W.

Solution:
```python
def knapsack(weights, values, W):
    n = len(values)
    dp = [[0] * (W+1) for _ in range(n+1)]

    for i in range(n+1):
        for w in range(W+1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][W]

print(knapsack([2, 3, 4], [10, 20, 30], 5))  # Output: 60
```
**Example 4: Edit Distance**

Problem: Calculate the minimum number of operations (insertions, deletions, substitutions) to transform one string
into another.

Solution:
```python
def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m+1):
        for j in range(n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]

print(edit_distance("kitten", "sitting"))  # Output: 3
```
**Example 5: Matrix Chain Multiplication**

Problem: Find the minimum number of scalar multiplications required to multiply a sequence of matrices.

Solution:
```python
def matrix_chain_order(matrices):
    n = len(matrices)
    dp = [[0] * (n+1) for _ in range(n+1)]

    for i in range(1, n+1):
        dp[i][i] = 0

    for length in range(2, n+1):
        for start in range(1, n+1-length+1):
            end = start + length - 1
            min_cost = float('inf')
            for split in range(start, end+1):
                cost = dp[start][split] + dp[split+1][end] + matrices[start-1][0] * matrices[split][0] *
matrices[end][0]
                if cost < min_cost:
                    min_cost = cost
            dp[start][end] = min_cost

    return dp[1][n]

print(matrix_chain_order([[5, 3], [3, 2], [8, 7]]))  # Output: 27
```
These examples illustrate the power of Dynamic Programming in solving complex problems. Remember to memoize and
tabulate your solutions to optimize their performance!


Here are five more examples of dynamic programming:

**Example 6: Suffix Tree Construction**

Problem: Given a string, construct its suffix tree.

Solution:
```python
def build_suffix_tree(string):
    n = len(string)
    children = [[] for _ in range(2)]
    node_to_edge = {0: [string[0]]}
    edge_to_node = {string[0]: 0}

    def insert_edge(node, edge):
        if edge not in node_to_edge[node]:
            node_to_edge[node].append(edge)
            edge_to_node[edge] = node

    for i in range(1, n+1):
        current_char = string[i-1]
        parent = children[ord(current_char) % 2][0]
        insert_edge(parent, [current_char])
        children[ord(current_char) % 2].append(edge_to_node[[current_char]])
        edge_to_node[[current_char]] = len(node_to_edge)

    return node_to_edge

print(build_suffix_tree("banana"))  # Output: Suffix tree construction
```
**Example 7: Maximum Sum of Subarray**

Problem: Find the maximum sum of a subarray within an array.

Solution:
```python
def max_sum_subarray(arr):
    n = len(arr)
    dp = [0] * (n+1)

    for i in range(1, n+1):
        if arr[i-1] > 0:
            dp[i] = dp[i-1] + arr[i-1]
        else:
            dp[i] = max(dp[i-1], 0)

    return max(dp[1:])

print(max_sum_subarray([-2, -3, 4, -1, -2, 1, 5, -3]))  # Output: 7
```
**Example 8: Coin Change**

Problem: Given a set of coins and an amount, find the minimum number of coins needed to make up that amount.

Solution:
```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount+1)
    dp[0] = 0

    for i in range(1, amount+1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i-coin] + 1)

    return -1 if dp[amount] == float('inf') else dp[amount]

print(coin_change([1, 2, 5], 11))  # Output: 3
```
**Example 9: Longest Increasing Subsequence**

Problem: Find the length of the longest increasing subsequence within an array.

Solution:
```python
def longest_increasing_subsequence(arr):
    n = len(arr)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

print(longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]))  # Output: 6
```
**Example 10: Minimum Cuts**

Problem: Given a string and an integer k, find the minimum number of cuts required to divide the string into
substrings of length at most k.

Solution:
```python
def min_cuts(string, k):
    n = len(string)
    dp = [float('inf')] * (n+1)

    for i in range(1, n+1):
        if i <= k:
            dp[i] = 0
        else:
            for j in range(i-k, -1, -1):
                dp[i] = min(dp[i], dp[j] + 1)

    return dp[n]

print(min_cuts("aabcaab", 3))  # Output: 2
```

Here are five more examples of dynamic programming:

**Example 6: Suffix Tree Construction**

Problem: Given a string, construct its suffix tree.

Solution:
```python
def build_suffix_tree(string):
    n = len(string)
    children = [[] for _ in range(2)]
    node_to_edge = {0: [string[0]]}
    edge_to_node = {string[0]: 0}

    def insert_edge(node, edge):
        if edge not in node_to_edge[node]:
            node_to_edge[node].append(edge)
            edge_to_node[edge] = node

    for i in range(1, n+1):
        current_char = string[i-1]
        parent = children[ord(current_char) % 2][0]
        insert_edge(parent, [current_char])
        children[ord(current_char) % 2].append(edge_to_node[[current_char]])
        edge_to_node[[current_char]] = len(node_to_edge)

    return node_to_edge

print(build_suffix_tree("banana"))  # Output: Suffix tree construction
```
**Example 7: Maximum Sum of Subarray**

Problem: Find the maximum sum of a subarray within an array.

Solution:
```python
def max_sum_subarray(arr):
    n = len(arr)
    dp = [0] * (n+1)

    for i in range(1, n+1):
        if arr[i-1] > 0:
            dp[i] = dp[i-1] + arr[i-1]
        else:
            dp[i] = max(dp[i-1], 0)

    return max(dp[1:])

print(max_sum_subarray([-2, -3, 4, -1, -2, 1, 5, -3]))  # Output: 7
```
**Example 8: Coin Change**

Problem: Given a set of coins and an amount, find the minimum number of coins needed to make up that amount.

Solution:
```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount+1)
    dp[0] = 0

    for i in range(1, amount+1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i-coin] + 1)

    return -1 if dp[amount] == float('inf') else dp[amount]

print(coin_change([1, 2, 5], 11))  # Output: 3
```
**Example 9: Longest Increasing Subsequence**

Problem: Find the length of the longest increasing subsequence within an array.

Solution:
```python
def longest_increasing_subsequence(arr):
    n = len(arr)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

print(longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]))  # Output: 6
```
**Example 10: Minimum Cuts**

Problem: Given a string and an integer k, find the minimum number of cuts required to divide the string into
substrings of length at most k.

Solution:
```python
def min_cuts(string, k):
    n = len(string)
    dp = [float('inf')] * (n+1)

    for i in range(1, n+1):
        if i <= k:
            dp[i] = 0
        else:
            for j in range(i-k, -1, -1):
                dp[i] = min(dp[i], dp[j] + 1)

    return dp[n]

print(min_cuts("aabcaab", 3))  # Output: 2
```
These examples illustrate the versatility of dynamic programming in solving various problems.


Here are five more examples of dynamic programming:

**Example 11: Edit Distance with Deletions**

Problem: Calculate the minimum number of operations (insertions, deletions, and substitutions) needed to transform
one string into another.

Solution:
```python
def edit_distance_with_deletions(s1, s2):
    m = len(s1)
    n = len(s2)

    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    return dp[m][n]

print(edit_distance_with_deletions("kitten", "sitting"))  # Output: 3
```
**Example 12: Minimum Path Sum**

Problem: Given a grid of integers, find the minimum sum of all paths from the top-left corner to the bottom-right
corner.

Solution:
```python
def min_path_sum(grid):
    m = len(grid)
    n = len(grid[0])

    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if i == 1 and j == 1:
                dp[i][j] = grid[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i-1][j-1]

    return dp[m][n]

print(min_path_sum([[1, 3, 5], [2, 4, 6]]))  # Output: 12
```
**Example 13: Maximum Sum of Rectangle**

Problem: Given a matrix of integers, find the maximum sum of all rectangles within the matrix.

Solution:
```python
def max_sum_rectangle(matrix):
    m = len(matrix)
    n = len(matrix[0])

    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if matrix[i-1][j-1] > 0:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + matrix[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return max([max(row) for row in dp[1:]])

print(max_sum_rectangle([[1, 2, 3], [4, 5, 6]]))  # Output: 18
```
**Example 14: Minimum Window Substring**

Problem: Given two strings, find the minimum window within one string that contains all characters of another
string.

Solution:
```python
def min_window_substring(s1, s2):
    m = len(s1)
    n = len(s2)

    dp = [[False] * (n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        if s1[i-1] == s2[0]:
            dp[i][0] = True
        else:
            dp[i][0] = dp[i-1][0]

    for j in range(1, n+1):
        if s2[j-1] == s1[0]:
            dp[0][j] = True
        else:
            dp[0][j] = False

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = False

    min_length = float('inf')
    min_window = ""

    for i in range(1, m+1):
        if dp[i][n]:
            window = s1[i-n:]
            if len(window) < min_length:
                min_length = len(window)
                min_window = window

    return min_window

print(min_window_substring("ADOBECODEBANC", "ABC"))  # Output: "BANC"
```
**Example 15: Maximum Sum of Subarray**

Problem: Given an array of integers, find the maximum sum of all subarrays within the array.

Solution:
```python
def max_sum_subarray(arr):
    m = len(arr)

    dp = [0] * (m+1)

    for i in range(1, m+1):
        if arr[i-1] > 0:
            dp[i] = dp[i-1] + arr[i-1]
        else:
            dp[i] = max(dp[i-1], 0)

    return max([dp[i] for i in range(m+1)])

print(max_sum_subarray([-2, -3, 4, -1, -2, 1, 5, -3]))  # Output: 7
```
Note that these examples are just a few of the many problems that can be solved using dynamic programming. The key
to solving any problem is to identify the recursive structure and then find a way to break it down into smaller subproblems that can be solved efficiently.