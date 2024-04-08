//1.Reverse a linked list:

class Node {
    constructor(data) {
        this.data = data;
        this.next = null;
    }
}

class LinkedList {
    constructor() {
        this.head = null;
    }

    reverse() {
        let prev = null;
        let current = this.head;
        let next = null;
        while (current) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        this.head = prev;
    }
}

//2.Find the middle element of a linked list:

function findMiddle(head) {
    let slow = head;
    let fast = head;
    while (fast && fast.next) {
        slow = slow.next;
        fast = fast.next.next;
    }
    return slow.data;
}


//3.Implement a stack using arrays:

class Stack {
    constructor() {
        this.items = [];
    }

    push(element) {
        this.items.push(element);
    }

    pop() {
        if (this.items.length === 0) return "Underflow";
        return this.items.pop();
    }

    peek() {
        return this.items[this.items.length - 1];
    }

    isEmpty() {
        return this.items.length === 0;
    }
}


//4.Implement a queue using arrays:

class Queue {
    constructor() {
        this.items = [];
    }

    enqueue(element) {
        this.items.push(element);
    }

    dequeue() {
        if (this.isEmpty()) return "Underflow";
        return this.items.shift();
    }

    front() {
        if (this.isEmpty()) return "Queue is empty";
        return this.items[0];
    }

    isEmpty() {
        return this.items.length === 0;
    }
}


//5.Find the factorial of a number using recursion:

function factorial(n) {
    if (n === 0 || n === 1) return 1;
    return n * factorial(n - 1);
}


//6.Implement binary search in an array:

function binarySearch(arr, target) {
    let low = 0;
    let high = arr.length - 1;
    while (low <= high) {
        let mid = Math.floor((low + high) / 2);
        if (arr[mid] === target) return mid;
        else if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}


//7.Find the largest element in an array:

function findLargest(arr) {
    let max = arr[0];
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}



//8.Implement merge sort:

function mergeSort(arr) {
    if (arr.length <= 1) return arr;
    
    const middle = Math.floor(arr.length / 2);
    const left = mergeSort(arr.slice(0, middle));
    const right = mergeSort(arr.slice(middle));

    return merge(left, right);
}

function merge(left, right) {
    let result = [];
    let leftIndex = 0;
    let rightIndex = 0;

    while (leftIndex < left.length && rightIndex < right.length) {
        if (left[leftIndex] < right[rightIndex]) {
            result.push(left[leftIndex]);
            leftIndex++;
        } else {
            result.push(right[rightIndex]);
            rightIndex++;
        }
    }

    return result.concat(left.slice(leftIndex)).concat(right.slice(rightIndex));
}


//9.Implement quick sort:

function quickSort(arr) {
    if (arr.length <= 1) return arr;

    const pivot = arr[0];
    const left = [];
    const right = [];

    for (let i = 1; i < arr.length; i++) {
        if (arr[i] < pivot) left.push(arr[i]);
        else right.push(arr[i]);
    }

    return quickSort(left).concat(pivot, quickSort(right));
}


//10.Detect a cycle in a linked list:

function hasCycle(head) {
    let slow = head;
    let fast = head;

    while (fast && fast.next) {
        slow = slow.next;
        fast = fast.next.next;

        if (slow === fast) return true;
    }

    return false;
}


//11.Find the intersection point of two linked lists:

function getIntersectionNode(headA, headB) {
    let pointerA = headA;
    let pointerB = headB;

    while (pointerA !== pointerB) {
        pointerA = pointerA ? pointerA.next : headB;
        pointerB = pointerB ? pointerB.next : headA;
    }

    return pointerA;
}


//12.Check if a binary tree is a binary search tree (BST):

function isBST(root, min = -Infinity, max = Infinity) {
    if (!root) return true;

    if (root.data <= min || root.data >= max) return false;

    return isBST(root.left, min, root.data) && isBST(root.right, root.data, max);
}


//13.Print all leaf nodes of a binary tree:

function printLeafNodes(root) {
    if (!root) return;

    if (!root.left && !root.right) console.log(root.data);

    printLeafNodes(root.left);
    printLeafNodes(root.right);
}


//14.Reverse a binary tree:

function invertTree(root) {
    if (!root) return null;

    const left = invertTree(root.left);
    const right = invertTree(root.right);

    root.left = right;
    root.right = left;

    return root;
}


//15.Find the height of a binary tree:

function getHeight(root) {
    if (!root) return -1;

    const leftHeight = getHeight(root.left);
    const rightHeight = getHeight(root.right);

    return 1 + Math.max(leftHeight, rightHeight);
}


//16.Implement depth-first search (DFS) on a graph:

function dfs(graph, start, visited = new Set()) {
    visited.add(start);
    console.log(start);
    for (const neighbor of graph[start]) {
        if (!visited.has(neighbor)) {
            dfs(graph, neighbor, visited);
        }
    }
}


//17.Implement breadth-first search (BFS) on a graph:

function bfs(graph, start) {
    const queue = [start];
    const visited = new Set(queue);
    while (queue.length) {
        const vertex = queue.shift();
        console.log(vertex);
        for (const neighbor of graph[vertex]) {
            if (!visited.has(neighbor)) {
                visited.add(neighbor);
                queue.push(neighbor);
            }
        }
    }
}


//18.Check if a graph is connected:

function isConnected(graph) {
    const visited = new Set();
    dfs(graph, Object.keys(graph)[0], visited);
    return visited.size === Object.keys(graph).length;
}


//19.Implement Dijkstra's algorithm for shortest path:

function dijkstra(graph, start) {
    const distances = {};
    const visited = new Set();
    for (const vertex in graph) {
        distances[vertex] = Infinity;
    }
    distances[start] = 0;

    while (true) {
        let current = null;
        for (const vertex in graph) {
            if (!visited.has(vertex) && (current === null || distances[vertex] < distances[current])) {
                current = vertex;
            }
        }

        if (current === null) break;

        visited.add(current);

        for (const neighbor in graph[current]) {
            const weight = graph[current][neighbor];
            const totalDistance = distances[current] + weight;
            if (totalDistance < distances[neighbor]) {
                distances[neighbor] = totalDistance;
            }
        }
    }

    return distances;
}


//20.Implement Prim's algorithm for minimum spanning tree:

function prim(graph) {
    const MST = new Set();
    const visited = new Set();
    const startNode = Object.keys(graph)[0];
    MST.add(startNode);
    visited.add(startNode);
    
    while (MST.size < Object.keys(graph).length) {
        let minEdge = [null, null, Infinity];
        for (const node of MST) {
            for (const neighbor in graph[node]) {
                if (!visited.has(neighbor) && graph[node][neighbor] < minEdge[2]) {
                    minEdge = [node, neighbor, graph[node][neighbor]];
                }
            }
        }
        MST.add(minEdge[1]);
        visited.add(minEdge[1]);
    }
    
    return MST;
}


//21.Implement Kruskal's algorithm for minimum spanning tree:

function kruskal(graph) {
    const edges = [];
    for (const node in graph) {
        for (const neighbor in graph[node]) {
            edges.push([node, neighbor, graph[node][neighbor]]);
        }
    }
    edges.sort((a, b) => a[2] - b[2]);

    const MST = new Set();
    const uf = new UnionFind();

    for (const node in graph) {
        uf.makeSet(node);
    }

    for (const edge of edges) {
        const [node1, node2, weight] = edge;
        if (uf.find(node1) !== uf.find(node2)) {
            MST.add(node1);
            MST.add(node2);
            uf.union(node1, node2);
        }
    }

    return MST;
}

class UnionFind {
    constructor() {
        this.parent = {};
    }

    makeSet(node) {
        this.parent[node] = node;
    }

    find(node) {
        if (this.parent[node] === node) return node;
        return this.find(this.parent[node]);
    }

    union(node1, node2) {
        const root1 = this.find(node1);
        const root2 = this.find(node2);
        this.parent[root2] = root1;
    }
}

//22.Find the longest common subsequence of two strings:

function longestCommonSubsequence(text1, text2) {
    const m = text1.length;
    const n = text2.length;
    const dp = Array.from({ length: m + 1 }, () => Array.from({ length: n + 1 }, () => 0));

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (text1[i - 1] === text2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    let i = m, j = n;
    let result = '';
    while (i > 0 && j > 0) {
        if (text1[i - 1] === text2[j - 1]) {
            result = text1[i - 1] + result;
            i--;
            j--;
        } else if (dp[i - 1][j] > dp[i][j - 1]) {
            i--;
        } else {
            j--;
        }
    }

    return result;
}

//23.Find the longest increasing subsequence of an array:

function longestIncreasingSubsequence(nums) {
    const dp = Array(nums.length).fill(1);
    let maxLength = 1;

    for (let i = 1; i < nums.length; i++) {
        for (let j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = Math.max(dp[i], dp[j] + 1);
                maxLength = Math.max(maxLength, dp[i]);
            }
        }
    }

    return maxLength;
}

//24.Implement the Knuth–Morris–Pratt (KMP) algorithm for string matching:

function kmp(text, pattern) {
    const lps = computeLPSArray(pattern);
    let i = 0, j = 0;
    const indices = [];

    while (i < text.length) {
        if (text[i] === pattern[j]) {
            i++;
            j++;
        }

        if (j === pattern.length) {
            indices.push(i - j);
            j = lps[j - 1];
        } else if (i < text.length && text[i] !== pattern[j]) {
            if (j !== 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }

    return indices;
}

function computeLPSArray(pattern) {
    const lps = Array(pattern.length).fill(0);
    let len = 0, i = 1;

    while (i < pattern.length) {
        if (pattern[i] === pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len !== 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }

    return lps;
}

//25.Implement the Rabin-Karp algorithm for string matching:

function rabinKarp(text, pattern) {
    const prime = 101;
    const textLength = text.length;
    const patternLength = pattern.length;
    const patternHash = hash(pattern, patternLength);
    let textHash = hash(text, patternLength);
    const indices = [];

    for (let i = 0; i <= textLength - patternLength; i++) {
        if (textHash === patternHash && text.substring(i, i + patternLength) === pattern) {
            indices.push(i);
        }
        if (i < textLength - patternLength) {
            textHash = recalculateHash(text, i, i + patternLength, textHash, patternLength, prime);
        }
    }

    return indices;
}

function hash(str, length) {
    let hashValue = 0;
    for (let i = 0; i < length; i++) {
        hashValue += str.charCodeAt(i) * Math.pow(101, i);
    }
    return hashValue;
}

function recalculateHash(str, oldIndex, newIndex, oldHash, patternLength, prime) {
    let newHash = oldHash - str.charCodeAt(oldIndex);
    newHash /= 101;
    newHash += str.charCodeAt(newIndex) * Math.pow(101, patternLength - 1);
    return newHash;
}

//26.Check if a string is a palindrome:

function isPalindrome(str) {
    let left = 0;
    let right = str.length - 1;
    while (left < right) {
        if (str[left] !== str[right]) return false;
        left++;
        right--;
    }
    return true;
}


//27.Check if two strings are anagrams of each other:

function areAnagrams(str1, str2) {
    if (str1.length !== str2.length) return false;

    const charCount = {};

    for (let char of str1) {
        charCount[char] = (charCount[char] || 0) + 1;
    }

    for (let char of str2) {
        if (!charCount[char]) return false;
        charCount[char]--;
    }

    return true;
}


//28.Find the next greater element in an array:

function nextGreaterElement(nums) {
    const stack = [];
    const result = new Array(nums.length).fill(-1);
    
    for (let i = 0; i < nums.length; i++) {
        while (stack.length && nums[i] > nums[stack[stack.length - 1]]) {
            result[stack.pop()] = nums[i];
        }
        stack.push(i);
    }
    
    return result;
}


//29.Find the kth smallest/largest element in an array:

function kthSmallest(nums, k) {
    nums.sort((a, b) => a - b);
    return nums[k - 1];
}

function kthLargest(nums, k) {
    nums.sort((a, b) => b - a);
    return nums[k - 1];
}


//30.Find the median of two sorted arrays:

function findMedianSortedArrays(nums1, nums2) {
    const merged = mergeSortedArrays(nums1, nums2);
    const n = merged.length;
    if (n % 2 === 0) {
        return (merged[n / 2 - 1] + merged[n / 2]) / 2;
    } else {
        return merged[Math.floor(n / 2)];
    }
}

function mergeSortedArrays(nums1, nums2) {
    const merged = [];
    let i = 0, j = 0;
    while (i < nums1.length && j < nums2.length) {
        if (nums1[i] < nums2[j]) {
            merged.push(nums1[i]);
            i++;
        } else {
            merged.push(nums2[j]);
            j++;
        }
    }
    while (i < nums1.length) {
        merged.push(nums1[i]);
        i++;
    }
    while (j < nums2.length) {
        merged.push(nums2[j]);
        j++;
    }
    return merged;
}

//31.Implement a trie (prefix tree):

class TrieNode {
    constructor() {
        this.children = {};
        this.isEndOfWord = false;
    }
}

class Trie {
    constructor() {
        this.root = new TrieNode();
    }

    insert(word) {
        let node = this.root;
        for (let char of word) {
            if (!node.children[char]) {
                node.children[char] = new TrieNode();
            }
            node = node.children[char];
        }
        node.isEndOfWord = true;
    }

    search(word) {
        let node = this.root;
        for (let char of word) {
            if (!node.children[char]) {
                return false;
            }
            node = node.children[char];
        }
        return node.isEndOfWord;
    }

    startsWith(prefix) {
        let node = this.root;
        for (let char of prefix) {
            if (!node.children[char]) {
                return false;
            }
            node = node.children[char];
        }
        return true;
    }
}

//32.Find all subsets of a set:

function subsets(nums) {
    const result = [[]];
    for (const num of nums) {
        const n = result.length;
        for (let i = 0; i < n; i++) {
            const subset = result[i].slice();
            subset.push(num);
            result.push(subset);
        }
    }
    return result;
}

//33.Find all permutations of a string:

function permute(str) {
    const result = [];
    permuteHelper(str.split(''), 0, result);
    return result;
}

function permuteHelper(arr, index, result) {
    if (index === arr.length) {
        result.push(arr.join(''));
    }
    for (let i = index; i < arr.length; i++) {
        [arr[index], arr[i]] = [arr[i], arr[index]];
        permuteHelper(arr, index + 1, result);
        [arr[index], arr[i]] = [arr[i], arr[index]];
    }
}


//34.Implement the Josephus Problem:

function josephus(n, k) {
    let survivor = 0;
    for (let i = 2; i <= n; i++) {
        survivor = (survivor + k) % i;
    }
    return survivor + 1;
}


//35.Implement an LRU (Least Recently Used) Cache:

class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();
    }

    get(key) {
        if (!this.cache.has(key)) return -1;
        const value = this.cache.get(key);
        this.cache.delete(key);
        this.cache.set(key, value);
        return value;
    }

    put(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.capacity) {
            this.cache.delete(this.cache.keys().next().value);
        }
        this.cache.set(key, value);
    }
}


//36.Find the longest palindrome substring in a string

function longestPalindrome(s) {
    let longest = '';
    for (let i = 0; i < s.length; i++) {
        const oddPalindrome = expandAroundCenter(s, i, i);
        const evenPalindrome = expandAroundCenter(s, i, i + 1);
        const currentLongest = oddPalindrome.length > evenPalindrome.length ? oddPalindrome : evenPalindrome;
        if (currentLongest.length > longest.length) {
            longest = currentLongest;
        }
    }
    return longest;
}

function expandAroundCenter(s, left, right) {
    while (left >= 0 && right < s.length && s[left] === s[right]) {
        left--;
        right++;
    }
    return s.slice(left + 1, right);
}

//37.Implement a priority queue:

class PriorityQueue {
    constructor() {
        this.queue = [];
    }

    enqueue(element, priority) {
        this.queue.push({ element, priority });
        this.queue.sort((a, b) => a.priority - b.priority);
    }

    dequeue() {
        if (this.isEmpty()) return "Queue is empty";
        return this.queue.shift().element;
    }

    front() {
        if (this.isEmpty()) return "Queue is empty";
        return this.queue[0].element;
    }

    isEmpty() {
        return this.queue.length === 0;
    }
}


//38.Implement a hashmap (dictionary):

class HashMap {
    constructor() {
        this.map = {};
    }

    put(key, value) {
        this.map[key] = value;
    }

    get(key) {
        return this.map[key];
    }

    remove(key) {
        delete this.map[key];
    }

    contains(key) {
        return this.map.hasOwnProperty(key);
    }
}

//39.Count the number of inversions in an array:

function countInversions(nums) {
    let count = 0;
    mergeSort(nums);

    function mergeSort(arr) {
        if (arr.length <= 1) return arr;
        
        const middle = Math.floor(arr.length / 2);
        const left = mergeSort(arr.slice(0, middle));
        const right = mergeSort(arr.slice(middle));
    
        return merge(left, right);
    }

    function merge(left, right) {
        let result = [];
        let leftIndex = 0;
        let rightIndex = 0;

        while (leftIndex < left.length && rightIndex < right.length) {
            if (left[leftIndex] <= right[rightIndex]) {
                result.push(left[leftIndex]);
                leftIndex++;
            } else {
                result.push(right[rightIndex]);
                rightIndex++;
                count += left.length - leftIndex;
            }
        }

        return result.concat(left.slice(leftIndex)).concat(right.slice(rightIndex));
    }

    return count;
}
//40.Find the shortest path in a maze:

function shortestPath(maze, start, end) {
    const directions = [[-1, 0], [0, 1], [1, 0], [0, -1]];
    const queue = [[start, 0]];
    const visited = new Set([start.join('-')]);

    while (queue.length) {
        const [[x, y], dist] = queue.shift();
        if (x === end[0] && y === end[1]) return dist;

        for (const [dx, dy] of directions) {
            const nx = x + dx;
            const ny = y + dy;
            const key = nx + '-' + ny;
            if (nx >= 0 && nx < maze.length && ny >= 0 && ny < maze[0].length && maze[nx][ny] === 0 && !visited.has(key)) {
                queue.push([[nx, ny], dist + 1]);
                visited.add(key);
            }
        }
    }

    return -1;
}