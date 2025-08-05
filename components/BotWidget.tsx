"use client";
import { useState } from "react";
import { SendHorizonal, Bot, X } from "lucide-react";
import { Input } from "./ui/input";

interface Message {
  role: "user" | "bot";
  content: string;
}

export default function BotWidget() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "bot",
      content: "👋 Hi! I'm your DSAMate Bot. Ask me about DSA problems, algorithms, or data structures!",
    },
  ]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    const updatedMessages = [...messages, { role: "user" as const, content: userMessage }];
    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    try {
      // Simple local knowledge-based responses
      const botResponse = await generateDSAResponse(userMessage);
      setMessages([...updatedMessages, { role: "bot" as const, content: botResponse }]);
    } catch (error) {
      setMessages([
        ...updatedMessages,
        { role: "bot" as const, content: "⚠️ Sorry, I encountered an error. Please try asking about a specific DSA topic!" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed bottom-20 right-6 z-50">
      {/* Floating Bot Button */}
      <button
        onClick={() => setOpen(!open)}
        className="bg-blue-600 hover:bg-blue-700 w-16 h-16 rounded-full shadow-lg flex items-center justify-center transition-all duration-300 hover:scale-110"
      >
        <Bot size={28} className="text-white" />
      </button>

      {/* Chat Window */}
      {open && (
        <div className="mt-2 w-96 bg-white dark:bg-gray-800 p-4 rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 max-h-[70vh] overflow-hidden flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between mb-4 pb-2 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2">
              <Bot size={20} className="text-blue-600" />
              <h3 className="font-semibold text-gray-900 dark:text-white">DSAMate Bot</h3>
            </div>
            <button
              onClick={() => setOpen(false)}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <X size={18} />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-3 mb-4" style={{ maxHeight: "400px" }}>
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`p-3 rounded-lg text-sm whitespace-pre-line ${
                  msg.role === "user"
                    ? "bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100 ml-8"
                    : "bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 mr-8"
                }`}
              >
                {msg.content}
              </div>
            ))}
            {loading && (
              <div className="bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 p-3 rounded-lg text-sm mr-8">
                🤖 Thinking...
              </div>
            )}
          </div>

          {/* Input */}
          <div className="flex items-center gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about DSA problems..."
              className="flex-grow border border-gray-300 dark:border-gray-600 p-2 rounded-lg text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-700"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  handleSend();
                }
              }}
            />
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white p-2 rounded-lg transition-colors"
            >
              <SendHorizonal size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Simple DSA Knowledge Base Response Generator
async function generateDSAResponse(userMessage: string): Promise<string> {
  const message = userMessage.toLowerCase();

  // Stacks and Queues
  if (message.includes("stack") || message.includes("queue") || message.includes("parentheses") || message.includes("bracket")) {
    return `📚 **Stacks & Queues**

**Stack (LIFO - Last In, First Out):**
- Operations: push(), pop(), peek(), isEmpty()
- Use cases: Function calls, undo operations, expression evaluation

**Queue (FIFO - First In, First Out):**
- Operations: enqueue(), dequeue(), front(), isEmpty()
- Use cases: BFS, scheduling, buffering

**Common Stack Problems:**
\`\`\`python
# Valid Parentheses
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    return not stack
\`\`\`

**Key Problems:**
- Valid Parentheses, Next Greater Element
- Min Stack, Evaluate RPN
- Sliding Window Maximum (Deque)

**Time:** O(1) for basic operations

Which stack/queue problem are you solving?`;
  }

  // Heaps and Priority Queues
  if (message.includes("heap") || message.includes("priority queue") || message.includes("kth largest") || message.includes("top k")) {
    return `🏔️ **Heaps & Priority Queues**

**Heap Properties:**
- **Max Heap:** Parent ≥ children
- **Min Heap:** Parent ≤ children
- Complete binary tree structure

**Operations:**
- Insert: O(log n)
- Extract-Max/Min: O(log n)
- Peek: O(1)

**Python Implementation:**
\`\`\`python
import heapq

# Min heap (default)
heap = []
heapq.heappush(heap, 3)
min_val = heapq.heappop(heap)

# Max heap (negate values)
max_heap = []
heapq.heappush(max_heap, -3)
max_val = -heapq.heappop(max_heap)
\`\`\`

**Common Patterns:**
- **Top K Problems:** Use min heap of size K
- **Kth Largest:** Min heap
- **Merge K Sorted Lists:** Min heap
- **Median Finding:** Two heaps

**Key Problems:**
- Kth Largest Element, Top K Frequent
- Merge K Sorted Lists, Find Median

Which heap problem needs explanation?`;
  }

  // Sorting Algorithms
  if (message.includes("sort") || message.includes("quicksort") || message.includes("mergesort") || message.includes("bubble")) {
    return `🔄 **Sorting Algorithms**

**Comparison-Based Sorts:**

**1. Quick Sort:**
- Average: O(n log n), Worst: O(n²)
- In-place, not stable
- Pivot partitioning

**2. Merge Sort:**
- Always: O(n log n)
- Stable, requires O(n) extra space
- Divide and conquer

**3. Heap Sort:**
- Always: O(n log n)
- In-place, not stable
- Build heap + extract max

**Non-Comparison Sorts:**

**4. Counting Sort:**
- O(n + k) where k = range
- For integers in small range

**5. Radix Sort:**
- O(d × n) where d = digits
- For integers/strings

**When to use which sort?**
- General purpose: Quick Sort or Merge Sort
- Stability needed: Merge Sort
- Memory constrained: Heap Sort
- Small range integers: Counting Sort

Which sorting algorithm interests you?`;
  }

  // Backtracking
  if (message.includes("backtrack") || message.includes("permutation") || message.includes("combination") || message.includes("n-queens") || message.includes("sudoku")) {
    return `🔙 **Backtracking**

**Core Concept:** Try all possibilities, backtrack when invalid

**Template:**
\`\`\`python
def backtrack(path, choices):
    if is_valid_solution(path):
        result.append(path.copy())
        return
    
    for choice in choices:
        if is_valid_choice(choice, path):
            path.append(choice)  # Make choice
            backtrack(path, remaining_choices)
            path.pop()  # Backtrack
\`\`\`

**Classic Problems:**

**1. Generate Permutations:**
- All arrangements of elements
- Time: O(n! × n)

**2. Generate Combinations:**
- Choose k elements from n
- Time: O(C(n,k) × k)

**3. N-Queens:**
- Place N queens on N×N board
- No two queens attack each other

**4. Sudoku Solver:**
- Fill 9×9 grid with constraints
- Each row/column/box has 1-9

**Key Techniques:**
- Pruning: Skip invalid branches early
- State management: What to track?
- Choice representation: How to iterate?

Which backtracking problem are you tackling?`;
  }

  // Greedy Algorithms
  if (message.includes("greedy") || message.includes("interval") || message.includes("activity selection") || message.includes("huffman")) {
    return `💰 **Greedy Algorithms**

**Core Principle:** Make locally optimal choice at each step

**When Greedy Works:**
1. **Greedy Choice Property:** Local optimum leads to global optimum
2. **Optimal Substructure:** Problem can be broken into subproblems

**Classic Problems:**

**1. Activity Selection:**
\`\`\`python
def activity_selection(start, end):
    # Sort by end time
    activities = sorted(zip(start, end), key=lambda x: x[1])
    result = [activities[0]]
    last_end = activities[0][1]
    
    for s, e in activities[1:]:
        if s >= last_end:
            result.append((s, e))
            last_end = e
    return result
\`\`\`

**2. Fractional Knapsack:**
- Sort by value/weight ratio
- Take items greedily

**3. Huffman Coding:**
- Build optimal prefix-free codes
- Use min heap for tree construction

**4. Minimum Spanning Tree:**
- Kruskal's: Sort edges, use Union-Find
- Prim's: Start from vertex, grow tree

**Key Insight:** Prove greedy choice is safe!

Which greedy problem needs clarification?`;
  }

  // Bit Manipulation
  if (message.includes("bit") || message.includes("xor") || message.includes("binary") || message.includes("bitwise")) {
    return `🔢 **Bit Manipulation**

**Basic Operations:**
- **AND (&):** Both bits 1 → 1
- **OR (|):** Any bit 1 → 1  
- **XOR (^):** Different bits → 1
- **NOT (~):** Flip all bits
- **Left Shift (<<):** Multiply by 2^n
- **Right Shift (>>):** Divide by 2^n

**Useful Tricks:**
\`\`\`python
# Check if power of 2
n & (n - 1) == 0

# Get rightmost set bit
n & (-n)

# Clear rightmost set bit
n & (n - 1)

# Set bit at position i
n | (1 << i)

# Clear bit at position i
n & ~(1 << i)
\`\`\`

**XOR Properties:**
- a ^ a = 0
- a ^ 0 = a
- XOR is commutative and associative

**Common Problems:**
- **Single Number:** All appear twice except one
- **Missing Number:** Find missing from 1 to n
- **Counting Bits:** Count 1s in binary representation

Which bit manipulation concept needs explanation?`;
  }

  // Sliding Window
  if (message.includes("sliding window") || message.includes("substring") || message.includes("subarray") || message.includes("longest")) {
    return `🪟 **Sliding Window Technique**

**When to Use:**
- Contiguous subarray/substring problems
- Find optimal window (min/max length)
- All subarrays of size K

**Two Types:**

**1. Fixed Size Window:**
\`\`\`python
# Maximum sum subarray of size k
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    return max_sum
\`\`\`

**2. Variable Size Window:**
\`\`\`python
# Longest substring with at most k distinct chars
def longest_substring_k_distinct(s, k):
    char_count = {}
    left = max_len = 0
    
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_len = max(max_len, right - left + 1)
    return max_len
\`\`\`

**Common Problems:**
- Longest Substring Without Repeating Characters
- Minimum Window Substring
- Sliding Window Maximum

**Time:** Usually O(n) | **Space:** O(k) for tracking

Which sliding window problem are you working on?`;
  }

  // Union-Find
  if (message.includes("union find") || message.includes("disjoint set") || message.includes("connected components")) {
    return `🔗 **Union-Find (Disjoint Set Union)**

**Purpose:** Track connected components in dynamic graph

**Operations:**
- **Find:** Which set does element belong to?
- **Union:** Merge two sets

**Implementation with Path Compression & Union by Rank:**
\`\`\`python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
\`\`\`

**Time Complexity:**
- Both operations: O(α(n)) ≈ O(1) amortized
- α(n) is inverse Ackermann function

**Common Problems:**
- Number of Connected Components
- Redundant Connection
- Accounts Merge
- Minimum Spanning Tree (Kruskal's)

Which Union-Find application interests you?`;
  }

  // Two Sum
  if (message.includes("two sum") || message.includes("twosum")) {
    return `🎯 **Two Sum Problem**

**Problem:** Find two numbers in an array that add up to a target value.

**Approach:** Use HashMap to store complements
1. Iterate through array
2. For each number, calculate: complement = target - current_number  
3. Check if complement exists in HashMap
4. If yes → return indices, if no → store current number

**Time:** O(n) | **Space:** O(n)

**Example:**
Input: nums = [2,7,11,15], target = 9
Output: [0,1] (because 2 + 7 = 9)

💡 **Key Insight:** Trade space for time efficiency!

Need help with implementation or similar problems?`;
  }

  // Binary Search
  if (message.includes("binary search") || message.includes("search")) {
    return `🔍 **Binary Search**

**Concept:** Efficiently search in sorted array by dividing search space in half.

**Algorithm:**
1. Set left = 0, right = array.length - 1
2. While left ≤ right:
   - mid = left + (right - left) / 2
   - If target == array[mid] → return mid
   - If target < array[mid] → search left half
   - If target > array[mid] → search right half
3. Return -1 if not found

**Time:** O(log n) | **Space:** O(1)

**Key Points:**
- Array must be sorted
- Eliminates half search space each iteration
- Use left + (right - left) / 2 to avoid overflow

Want to see variations like rotated array search?`;
  }

  // Linked List
  if (message.includes("linked list") || message.includes("reverse")) {
    return `🔗 **Linked Lists**

**Key Concepts:**
- Dynamic structure with nodes (data + next pointer)
- No random access, must traverse from head
- Efficient insertion/deletion at known positions

**Common Patterns:**
1. **Two Pointers:** Fast/slow for cycle detection
2. **Dummy Node:** Simplifies edge cases
3. **Recursion:** Natural fit for linked lists

**Reverse Linked List:**
\`\`\`
prev = null, curr = head
while curr:
    next = curr.next
    curr.next = prev
    prev = curr
    curr = next
return prev
\`\`\`

**Time:** O(n) | **Space:** O(1)

Which linked list problem interests you?`;
  }

  // Dynamic Programming
  if (message.includes("dynamic programming") || message.includes("dp") || message.includes("climb")) {
    return `🧠 **Dynamic Programming**

**Core Idea:** Break complex problems into simpler subproblems, store results to avoid recomputation.

**When to Use:**
1. Optimal substructure (optimal solution contains optimal subproblems)
2. Overlapping subproblems (same subproblems solved multiple times)

**Common Patterns:**
- **1D DP:** Climbing stairs, house robber
- **2D DP:** Edit distance, LCS
- **Knapsack:** 0/1 knapsack, coin change

**Climbing Stairs Example:**
- Problem: Count ways to climb n stairs (1 or 2 steps)
- Recurrence: dp[i] = dp[i-1] + dp[i-2]
- Base: dp[1] = 1, dp[2] = 2

**Steps:**
1. Define state (what does dp[i] represent?)
2. Find recurrence relation
3. Identify base cases
4. Determine computation order

Which DP problem would you like to explore?`;
  }

  // Trees
  if (message.includes("tree") || message.includes("binary tree") || message.includes("bst")) {
    return `🌳 **Trees & Binary Search Trees**

**Tree Fundamentals:**
- Hierarchical structure: root, internal nodes, leaves
- Each node has ≤ one parent, ≥ zero children

**Binary Tree:**
- Each node has ≤ 2 children (left, right)
- Height = longest path from root to leaf

**BST Properties:**
- Left subtree < root < right subtree
- Inorder traversal gives sorted sequence
- Average O(log n) operations

**Traversals:**
- **Inorder:** Left → Root → Right (sorted for BST)
- **Preorder:** Root → Left → Right (copying)
- **Postorder:** Left → Right → Root (deletion)
- **Level Order:** BFS using queue

**Key Problems:**
- Maximum Depth, Validate BST
- Lowest Common Ancestor
- Level Order Traversal

What tree concept needs explanation?`;
  }

  // Arrays
  if (message.includes("array") || message.includes("string") || message.includes("kadane")) {
    return `📚 **Arrays & Strings**

**Array Techniques:**
1. **Two Pointers:** Sorted arrays, palindromes
2. **Sliding Window:** Subarray problems
3. **Prefix Sum:** Range sum queries
4. **HashMap:** Frequency counting, lookups

**String Techniques:**
1. **Character Arrays:** In-place modifications
2. **StringBuilder:** Efficient concatenation
3. **Pattern Matching:** KMP, Rabin-Karp
4. **Palindromes:** Expand around centers

**Maximum Subarray (Kadane's):**
\`\`\`
max_sum = current_sum = nums[0]
for i in range(1, len(nums)):
    current_sum = max(nums[i], current_sum + nums[i])
    max_sum = max(max_sum, current_sum)
\`\`\`

**Time:** O(n) | **Space:** O(1)

Which array/string problem do you need help with?`;
  }

  // Graphs
  if (message.includes("graph") || message.includes("dfs") || message.includes("bfs")) {
    return `📊 **Graphs**

**Graph Basics:**
- Vertices (nodes) connected by edges
- Directed vs Undirected, Weighted vs Unweighted

**Representations:**
1. **Adjacency List:** Space efficient O(V + E)
2. **Adjacency Matrix:** Fast lookup O(V²)

**Traversals:**
- **DFS:** Stack/recursion, pathfinding
- **BFS:** Queue, shortest path in unweighted graphs

**Common Problems:**
- Number of Islands (DFS/BFS on 2D grid)
- Clone Graph (DFS + HashMap)
- Course Schedule (Topological Sort)

**DFS Template:**
\`\`\`
def dfs(node, visited):
    if node in visited: return
    visited.add(node)
    for neighbor in graph[node]:
        dfs(neighbor, visited)
\`\`\`

What graph problem are you working on?`;
  }

  // General help
  if (message.includes("help") || message.includes("start") || message.includes("hello")) {
    return `👋 **Welcome to DSAMate Bot!**

I can help you with:

🎯 **Core Topics I Cover:**
- **Arrays & Strings** - Two pointers, sliding window, Kadane's algorithm
- **Linked Lists** - Reversal, cycle detection, two pointers
- **Trees & BST** - Traversals, validation, LCA
- **Graphs** - DFS, BFS, topological sort
- **Dynamic Programming** - Fibonacci, grid, string, knapsack patterns
- **Stacks & Queues** - Parentheses, monotonic stack, BFS
- **Heaps & Priority Queues** - Top K problems, median finding
- **Sorting Algorithms** - Quick, merge, heap, counting, radix
- **Backtracking** - Permutations, combinations, N-Queens, Sudoku
- **Greedy Algorithms** - Activity selection, MST, Huffman coding
- **Bit Manipulation** - XOR tricks, power of 2, single number
- **Sliding Window** - Fixed/variable size, substring problems
- **Union-Find** - Connected components, MST, dynamic connectivity
- **Binary Search** - Search variations, rotated arrays

💡 **How to Ask:**
- "Explain heap data structure"
- "How does backtracking work?"
- "Help with sliding window problems"
- "What are greedy algorithms?"
- "Show me bit manipulation tricks"

🚀 **I provide:**
- Step-by-step explanations
- Time/space complexity analysis
- Code examples and templates
- Problem patterns and variations
- When to use each technique

What DSA topic would you like to explore?`;
  }

  // Specific Problem Questions
  if (message.includes("reverse integer") || message.includes("palindrome number") || message.includes("roman to integer")) {
    return `🔢 **Common Coding Problems**

**Reverse Integer:**
\`\`\`python
def reverse(x):
    sign = -1 if x < 0 else 1
    x = abs(x)
    result = 0
    while x:
        result = result * 10 + x % 10
        x //= 10
    return sign * result if -2**31 <= sign * result <= 2**31 - 1 else 0
\`\`\`

**Palindrome Number:**
\`\`\`python
def is_palindrome(x):
    if x < 0: return False
    return str(x) == str(x)[::-1]
    # Or reverse half the number for O(1) space
\`\`\`

**Roman to Integer:**
- Use hashmap for symbol values
- If current < next, subtract; else add

Which specific problem needs detailed explanation?`;
  }

  // Matrix Problems
  if (message.includes("matrix") || message.includes("2d array") || message.includes("spiral") || message.includes("rotate matrix")) {
    return `🔲 **Matrix/2D Array Problems**

**Common Patterns:**

**1. Matrix Traversal:**
- Row by row, column by column
- Diagonal traversal
- Spiral traversal

**2. Spiral Matrix:**
\`\`\`python
def spiral_order(matrix):
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Left
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Up
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result
\`\`\`

**3. Rotate Matrix 90°:**
- Transpose + reverse each row
- Or rotate layer by layer

**4. Search in 2D Matrix:**
- Start from top-right or bottom-left
- Binary search on each row

Which matrix problem are you working on?`;
  }

  // String Algorithms
  if (message.includes("string algorithm") || message.includes("kmp") || message.includes("rabin karp") || message.includes("pattern matching")) {
    return `📝 **String Algorithms & Pattern Matching**

**1. KMP (Knuth-Morris-Pratt):**
- Preprocess pattern to build failure function
- Skip characters intelligently on mismatch
- Time: O(n + m)

**2. Rabin-Karp (Rolling Hash):**
\`\`\`python
def rabin_karp(text, pattern):
    base, mod = 256, 101
    m, n = len(pattern), len(text)
    
    # Calculate hash of pattern and first window
    pattern_hash = text_hash = 0
    h = pow(base, m-1) % mod
    
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % mod
        text_hash = (base * text_hash + ord(text[i])) % mod
    
    # Slide the pattern over text
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            if text[i:i+m] == pattern:
                return i
        
        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % mod
            if text_hash < 0:
                text_hash += mod
    
    return -1
\`\`\`

**3. Z-Algorithm:**
- Linear time string matching
- Builds Z-array for pattern preprocessing

**4. Manacher's Algorithm:**
- Find all palindromes in O(n) time
- Expand around centers efficiently

Which string algorithm interests you?`;
  }

  // Advanced Tree Problems
  if (message.includes("trie") || message.includes("segment tree") || message.includes("fenwick") || message.includes("binary indexed tree")) {
    return `🌲 **Advanced Tree Data Structures**

**1. Trie (Prefix Tree):**
\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
\`\`\`

**2. Segment Tree:**
- Range queries and updates in O(log n)
- Build: O(n), Query/Update: O(log n)
- Use cases: Range sum, range minimum/maximum

**3. Fenwick Tree (Binary Indexed Tree):**
- Efficient prefix sum queries
- Update and query in O(log n)
- Space efficient: O(n)

**Applications:**
- **Trie:** Autocomplete, word games, IP routing
- **Segment Tree:** Range queries, lazy propagation
- **Fenwick Tree:** Frequency counting, inversion counting

Which advanced tree structure needs explanation?`;
  }

  // Graph Algorithms
  if (message.includes("dijkstra") || message.includes("bellman ford") || message.includes("floyd warshall") || message.includes("shortest path")) {
    return `🗺️ **Graph Algorithms & Shortest Paths**

**1. Dijkstra's Algorithm:**
\`\`\`python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current_dist > distances[current]:
            continue
            
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
\`\`\`

**2. Bellman-Ford Algorithm:**
- Handles negative weights
- Detects negative cycles
- Time: O(VE)

**3. Floyd-Warshall Algorithm:**
- All-pairs shortest paths
- Dynamic programming approach
- Time: O(V³)

**4. A* Search:**
- Heuristic-based pathfinding
- Uses f(n) = g(n) + h(n)
- Optimal if heuristic is admissible

**When to use:**
- **Dijkstra:** Single-source, non-negative weights
- **Bellman-Ford:** Negative weights, cycle detection
- **Floyd-Warshall:** All-pairs, small graphs
- **A*:** Pathfinding with good heuristic

Which shortest path algorithm do you need help with?`;
  }

  // Advanced DP Patterns
  if (message.includes("lcs") || message.includes("edit distance") || message.includes("coin change") || message.includes("knapsack")) {
    return `💎 **Advanced Dynamic Programming Patterns**

**1. Longest Common Subsequence (LCS):**
\`\`\`python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
\`\`\`

**2. Edit Distance (Levenshtein):**
- Insert, delete, replace operations
- dp[i][j] = min cost to transform word1[:i] to word2[:j]

**3. Coin Change:**
\`\`\`python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
\`\`\`

**4. 0/1 Knapsack:**
- dp[i][w] = max value using first i items with weight ≤ w
- Choice: include item i or not

**5. Longest Increasing Subsequence:**
- O(n²) DP or O(n log n) with binary search
- Patience sorting approach

**Pattern Recognition:**
- **String DP:** LCS, edit distance, palindromes
- **Decision DP:** Knapsack, coin change, house robber
- **Path DP:** Unique paths, minimum path sum
- **Interval DP:** Matrix chain multiplication

Which DP pattern needs detailed explanation?`;
  }

  // Default response
  return `🤔 **I can help with that DSA topic!**

I specialize in:
- **Algorithms:** Sorting, searching, graph traversal, DP
- **Data Structures:** Arrays, linked lists, trees, graphs, heaps
- **Problem Solving:** Step-by-step approaches and optimizations

**Try asking:**
- "How do I solve Two Sum?"
- "Explain binary search algorithm"  
- "Help with linked list reversal"
- "What's the approach for [specific problem]?"
- "Show me matrix traversal patterns"
- "Explain string matching algorithms"
- "Help with advanced tree structures"
- "What are shortest path algorithms?"

**Popular topics:** Two Sum, Binary Search, Linked Lists, Trees, Dynamic Programming, Graphs, Matrix Problems, String Algorithms

What specific DSA problem or concept would you like me to explain?`;
}
