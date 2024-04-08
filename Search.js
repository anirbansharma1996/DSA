// [ Binary Search ]
const arr = [1, 2, 3, 4, 6, 7, 8, 9, 10];

function BinarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;
    let mid;

    while (left <= right) {
        mid = Math.floor((left + right) / 2);

        if (arr[mid] === target) {
            return mid;
        }
        if (arr[mid] > target) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return -1;
}

const res = BinarySearch(arr, 11);
console.log(res);

// [ Recursive Binary Search ]

function RecursiveBinarySearch(arr, left, right, t) {
  if (left <= right) {
    let mid = Math.floor((left + right) / 2);

    if (arr[mid] === t) {
      return mid;
    }
    if (arr[mid] > t) {
      return RecursiveBinarySearch(arr, left, mid - 1, t);
    } else {
      return RecursiveBinarySearch(arr, mid + 1, right, t);
    }
  }
  return -1;
}

let left = 0;
let right = arr.length - 1;
const ans = RecursiveBinarySearch(arr, left, right, 9);
console.log(ans);

//  [ Breadth-first search (BFS)  ]

// Breadth-first search (BFS) is an algorithm used for traversing or searching tree or graph data structures. It starts at the root node and explores all of the neighbor nodes at the present depth prior to moving on to the nodes at the next depth level. It visits all the vertices of a graph or tree level by level.

function bfs(graph, start) {
  let visited = {};
  let queue = [start];
  let result = [];

  visited[start] = true;

  while (queue.length > 0) {
    let vertex = queue.shift();
    result.push(vertex);

    graph[vertex].forEach((neighbor) => {
      if (!visited[neighbor]) {
        visited[neighbor] = true;
        queue.push(neighbor);
      }
    });
  }

  return result;
}

const graph = {
  A: ["B", "C"],
  B: ["A", "D", "E"],
  C: ["A", "F"],
  D: ["B"],
  E: ["B", "F"],
  F: ["C", "E"],
};

console.log(bfs(graph, "A"));



function Missing(arr) {
  let left = 0;
  let right = arr.length - 1;

  if (left <= right) {
   let mid = Math.floor((left + right) / 2);

    if (
      arr[mid] - mid > 1 &&
      (mid === 0 || arr[mid - 1] - (mid - 1) === 1)
    ) {
      return mid + 1;
    }
    if (arr[mid] - mid <= 1) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return arr.length+1;
}
const ans2 = Missing(arr);
console.log(ans2);

function RecursiveMissing(arr, left, right) {
  if (left <= right) {
    let mid = Math.floor((left + right) / 2);

    if (arr[mid] - mid > 1 && (mid === 0 || arr[mid - 1] - (mid - 1) === 1)) {
      return mid + 1;
    }
    if (arr[mid] - mid <= 1) {
      return RecursiveMissing(arr, mid + 1, right);
    } else {
      return RecursiveMissing(arr, left, mid - 1);
    }
  }
  return arr.length + 1;
}

let left3 = 0;
let right3 = arr.length - 1;
let ans3 = RecursiveMissing(arr, left3, right3);
console.log(ans3)
