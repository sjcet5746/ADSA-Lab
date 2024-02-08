#!/usr/bin/env python
# coding: utf-8

# In[1]:


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    # Function to insert a node into BST
    def insert(self, root, data):
        if root is None:
            return TreeNode(data)
        if data < root.data:
            root.left = self.insert(root.left, data)
        elif data > root.data:
            root.right = self.insert(root.right, data)
        return root

    # Function to delete a node from BST
    def delete(self, root, key):
        if root is None:
            return root
        if key < root.data:
            root.left = self.delete(root.left, key)
        elif key > root.data:
            root.right = self.delete(root.right, key)
        else:
            if root.left is None:
                temp = root.right
                root = None
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                return temp
            temp = self.min_value_node(root.right)
            root.data = temp.data
            root.right = self.delete(root.right, temp.data)
        return root

    # Function to find the minimum value node in a BST
    def min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    # Function to search a node in BST
    def search(self, root, key):
        if root is None or root.data == key:
            return root
        if key < root.data:
            return self.search(root.left, key)
        return self.search(root.right, key)

    # Function to display BST in inorder traversal
    def inorder_traversal(self, root):
        if root:
            self.inorder_traversal(root.left)
            print(root.data, end=" ")
            self.inorder_traversal(root.right)

    # Function to display BST in preorder traversal
    def preorder_traversal(self, root):
        if root:
            print(root.data, end=" ")
            self.preorder_traversal(root.left)
            self.preorder_traversal(root.right)

    # Function to display BST in postorder traversal
    def postorder_traversal(self, root):
        if root:
            self.postorder_traversal(root.left)
            self.postorder_traversal(root.right)
            print(root.data, end=" ")

if __name__ == "__main__":
    bst = BinarySearchTree()
    bst.root = bst.insert(bst.root, 50)
    bst.insert(bst.root, 30)
    bst.insert(bst.root, 20)
    bst.insert(bst.root, 40)
    bst.insert(bst.root, 70)
    bst.insert(bst.root, 60)
    bst.insert(bst.root, 80)

    print("Binary Search Tree in inorder traversal:", end=" ")
    bst.inorder_traversal(bst.root)
    print()

    bst.root = bst.delete(bst.root, 20)
    print("Binary Search Tree after deletion of 20 in inorder traversal:", end=" ")
    bst.inorder_traversal(bst.root)
    print()

    key = 30
    result = bst.search(bst.root, key)
    if result:
        print(f"Element {key} found in the BST.")
    else:
        print(f"Element {key} not found in the BST.")


# In[2]:


def binary_search(arr, key):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == key:
            return mid
        elif arr[mid] < key:
            left = mid + 1
        else:
            right = mid - 1
    return -1  # Key not found

# Example usage
arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
key = 23
index = binary_search(arr, key)
if index != -1:
    print(f"Element {key} found at index {index}.")
else:
    print(f"Element {key} not found.")


# In[14]:


class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None

class SplayTree:
    def __init__(self):
        self.root = None

    def zig(self, x):
        p = x.parent
        if p.left == x:
            p.left = x.right
            if x.right:
                x.right.parent = p
            x.right = p
        else:
            p.right = x.left
            if x.left:
                x.left.parent = p
            x.left = p
        x.parent = p.parent
        p.parent = x
        if x.parent:
            if x.parent.left == p:
                x.parent.left = x
            else:
                x.parent.right = x
        else:
            self.root = x

    def zigzig(self, x):
        p = x.parent
        g = p.parent
        if p.left == x:
            p.left = x.right
            if x.right:
                x.right.parent = p
            x.right = p
            g.left = p.right
            if p.right:
                p.right.parent = g
            p.right = g
        else:
            p.right = x.left
            if x.left:
                x.left.parent = p
            x.left = p
            g.right = p.left
            if p.left:
                p.left.parent = g
            p.left = g
        x.parent = g.parent
        g.parent = p
        p.parent = x
        if x.parent:
            if x.parent.left == g:
                x.parent.left = x
            else:
                x.parent.right = x
        else:
            self.root = x

    def zigzag(self, x):
        p = x.parent
        g = p.parent
        if p.right == x:
            p.right = x.left
            if x.left:
                x.left.parent = p
            x.left = p
            g.left = x.right
            if x.right:
                x.right.parent = g
            x.right = g
        else:
            p.left = x.right
            if x.right:
                x.right.parent = p
            x.right = p
            g.right = x.left
            if x.left:
                x.left.parent = g
            x.left = g
        x.parent = g.parent
        g.parent = x
        p.parent = x
        if x.parent:
            if x.parent.left == g:
                x.parent.left = x
            else:
                x.parent.right = x
        else:
            self.root = x

    def splay(self, x):
        while x.parent:
            p = x.parent
            g = p.parent
            if not g:
                self.zig(x)
            elif (g.left == p and p.left == x) or (g.right == p and p.right == x):
                self.zigzig(x)
            else:
                self.zigzag(x)

    def insert(self, key):
        if not self.root:
            self.root = Node(key)
            return
        x = self.root
        while x:
            p = x
            if key < x.key:
                x = x.left
            elif key > x.key:
                x = x.right
            else:
                return  # Key already exists
        if key < p.key:
            p.left = Node(key)
            p.left.parent = p
            self.splay(p.left)
        else:
            p.right = Node(key)
            p.right.parent = p
            self.splay(p.right)

    def search(self, key):
        if not self.root:
            return None
        x = self.root
        while x:
            if key == x.key:
                self.splay(x)
                return x
            elif key < x.key:
                x = x.left
            else:
                x = x.right
        return None

    def delete(self, key):
        node = self.search(key)
        if not node:
            return
        if node.left and node.right:
            self.splay(node.left)
            self.splay(node.right)
            l = node.left
            r = node.right
            l.right = r
            r.parent = l
            self.root = l
        elif node.left:
            l = node.left
            self.root = l
        elif node.right:
            r = node.right
            self.root = r
        else:
            self.root = None

    def inorder_traversal(self, root):
        if root:
            self.inorder_traversal(root.left)
            print(root.key, end=" ")
            self.inorder_traversal(root.right)

    def inorder(self):
        self.inorder_traversal(self.root)
        print()

# Example usage
splay_tree = SplayTree()
splay_tree.insert(50)
splay_tree.insert(30)
splay_tree.insert(70)
splay_tree.insert(20)
splay_tree.insert(40)
splay_tree.insert(60)
splay_tree.insert(80)

print("Inorder traversal before search:")
splay_tree.inorder()

print("Search 40:")
splay_tree.search(40)
print("Inorder traversal after search (40 should be at root):")
splay_tree.inorder()

print("Delete 40:")
splay_tree.delete(40)
print("Inorder traversal after delete (40 should be removed):")
splay_tree.inorder()


# In[4]:


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # Finding the mid of the array
        left_half = arr[:mid]  # Dividing the array elements into two halves
        right_half = arr[mid:]

        merge_sort(left_half)  # Sorting the first half
        merge_sort(right_half)  # Sorting the second half

        i = j = k = 0

        # Copy data to temporary arrays left_half[] and right_half[]
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

# Example usage
if __name__ == "__main__":
    arr = [12, 11, 13, 5, 6, 7]
    print("Given array is", arr)

    merge_sort(arr)

    print("Sorted array is", arr)


# In[5]:


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less_than_pivot = [x for x in arr[1:] if x <= pivot]
        greater_than_pivot = [x for x in arr[1:] if x > pivot]
        return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)

# Example usage
if __name__ == "__main__":
    arr = [10, 7, 8, 9, 1, 5]
    print("Given array:", arr)

    sorted_arr = quick_sort(arr)

    print("Sorted array:", sorted_arr)


# In[6]:


def knapsack_greedy(values, weights, capacity):
    n = len(values)
    # Calculate value-to-weight ratios
    value_weight_ratio = [(values[i] / weights[i], i) for i in range(n)]
    # Sort items based on their value-to-weight ratio in descending order
    value_weight_ratio.sort(reverse=True)

    total_value = 0
    remaining_capacity = capacity

    for ratio, i in value_weight_ratio:
        if weights[i] <= remaining_capacity:
            total_value += values[i]
            remaining_capacity -= weights[i]
        else:
            total_value += values[i] * (remaining_capacity / weights[i])
            break

    return total_value

# Example usage
if __name__ == "__main__":
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    max_value = knapsack_greedy(values, weights, capacity)
    print("Maximum value obtained from the knapsack:", max_value)


# In[7]:


import sys

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

    def printMST(self, parent):
        print("Edge \tWeight")
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.graph[i][parent[i]])

    def minKey(self, key, mstSet):
        min = sys.maxsize
        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
        return min_index

    def primMST(self):
        key = [sys.maxsize] * self.V
        parent = [None] * self.V
        key[0] = 0
        mstSet = [False] * self.V
        parent[0] = -1

        for cout in range(self.V):
            u = self.minKey(key, mstSet)
            mstSet[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        self.printMST(parent)


g = Graph(5)
g.graph = [
    [0, 2, 0, 6, 0],
    [2, 0, 3, 8, 5],
    [0, 3, 0, 0, 7],
    [6, 8, 0, 0, 9],
    [0, 5, 7, 9, 0]
]

g.primMST()


# In[8]:


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def KruskalMST(self):
        result = []
        i, e = 0, 0

        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)

        minimumCost = 0
        print("Minimum Spanning Tree:")
        for u, v, weight in result:
            print("%d - %d : %d" % (u, v, weight))
            minimumCost += weight
        print("Minimum Cost:", minimumCost)


g = Graph(4)
g.add_edge(0, 1, 10)
g.add_edge(0, 2, 6)
g.add_edge(0, 3, 5)
g.add_edge(1, 3, 15)
g.add_edge(2, 3, 4)

g.KruskalMST()


# In[9]:


import sys

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

    def printSolution(self, dist):
        print("Vertex \t Distance from Source")
        for node in range(self.V):
            print(node, "\t", dist[node])

    def minDistance(self, dist, sptSet):
        min = sys.maxsize
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
        return min_index

    def dijkstra(self, src):
        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):
            u = self.minDistance(dist, sptSet)
            sptSet[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]

        self.printSolution(dist)


g = Graph(9)
g.graph = [
    [0, 4, 0, 0, 0, 0, 0, 8, 0],
    [4, 0, 8, 0, 0, 0, 0, 11, 0],
    [0, 8, 0, 7, 0, 4, 0, 0, 2],
    [0, 0, 7, 0, 9, 14, 0, 0, 0],
    [0, 0, 0, 9, 0, 10, 0, 0, 0],
    [0, 0, 4, 14, 10, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 1, 6],
    [8, 11, 0, 0, 0, 0, 1, 0, 7],
    [0, 0, 2, 0, 0, 0, 6, 7, 0]
]

g.dijkstra(0)


# In[10]:


def job_sequencing(jobs):
    jobs.sort(key=lambda x: x[2], reverse=True)  # Sort jobs based on profit in descending order
    max_deadline = max(jobs, key=lambda x: x[1])[1]  # Find the maximum deadline
    result = [None] * max_deadline  # Initialize result array with None

    for job in jobs:
        deadline = job[1]
        while deadline > 0:
            if result[deadline - 1] is None:  # Check if the slot is empty
                result[deadline - 1] = job[0]  # Assign the job to the slot
                break
            deadline -= 1

    return [job for job in result if job is not None]  # Remove None values


# Example usage
jobs = [('a', 2, 100),
        ('b', 1, 19),
        ('c', 2, 27),
        ('d', 1, 25),
        ('e', 3, 15)]

sequence = job_sequencing(jobs)
print("The sequence of jobs:", sequence)


# In[11]:


def knapSack(W, wt, val, n):
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    return K[n][W]

# Example usage
val = [60, 100, 120]
wt = [10, 20, 30]
W = 50
n = len(val)

print("Maximum value that can be obtained:", knapSack(W, wt, val, n))


# In[12]:


def sum_of_subsets(s, k, r, x, w, target_sum):
    x[k] = 1

    if s + w[k] == target_sum:
        print_subset(x, w)
    elif s + w[k] + w[k + 1] <= target_sum:
        sum_of_subsets(s + w[k], k + 1, r - w[k], x, w, target_sum)

    if s + r - w[k] >= target_sum and s + w[k + 1] <= target_sum:
        x[k] = 0
        sum_of_subsets(s, k + 1, r - w[k], x, w, target_sum)


def print_subset(x, w):
    subset = []
    for i in range(len(x)):
        if x[i] == 1:
            subset.append(w[i])
    print("Subset:", subset)


def main():
    n = int(input("Enter number of elements: "))
    w = []

    print("Enter the elements:")
    for i in range(n):
        w.append(int(input()))

    target_sum = int(input("Enter the required sum: "))

    w.sort()  # Sort the array to optimize backtracking

    x = [0] * n
    total = sum(w)

    print("\nSubsets with sum equal to", target_sum, "are:")
    sum_of_subsets(0, 0, total, x, w, target_sum)


if __name__ == "__main__":
    main()


# In[13]:


def is_safe(board, row, col, N):
    # Check if there is a queen in the current column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper diagonal on right side
    for i, j in zip(range(row, -1, -1), range(col, N)):
        if board[i][j] == 1:
            return False

    return True


def solve_n_queens_util(board, row, N):
    if row == N:
        return True

    for col in range(N):
        if is_safe(board, row, col, N):
            board[row][col] = 1
            if solve_n_queens_util(board, row + 1, N):
                return True
            board[row][col] = 0  # Backtrack

    return False


def solve_n_queens(N):
    board = [[0] * N for _ in range(N)]

    if not solve_n_queens_util(board, 0, N):
        print("Solution does not exist")
        return

    print_solution(board)


def print_solution(board):
    for row in board:
        print(" ".join(map(str, row)))


# Example usage
N = 8
solve_n_queens(N)


# In[ ]:




