var arr = [3, 1, 44, 5, 23];
// common swap function
function swap(n, i, j) {
  // console.log(n,i,j)
  let temp = n[i];
  n[i] = n[j];
  n[j] = temp;
}

// Selection Sort
function SelectionSort(n) {
  for (let i = 0; i < n.length - 1; i++) {
    let min = i;
    for (let j = i + 1; j < n.length; j++) {
      if (n[j] < n[min]) {
        swap(n, min, j);
      }
    }
  }
  console.log('selection sort ->',n);
}
SelectionSort(arr)

//Bubble Sort
function BubbleSort(n) {
  for (let i = 0; i < n.length - 1; i++) {
    for (let j = 0; j < n.length - i - 1; j++) {
      if (n[j] > n[j + 1]) {
        swap(n, j + 1, j);
      }
    }
  }
  console.log('bubble sort ->',n);
}
BubbleSort(arr)

//Quick sort

function QuickSort(n) {
  if (n.length <= 1) {
    return n;
  }

  let pivot = n[Math.floor(n.length / 2)];
  let left = [];
  let right = [];

  for (let i = 0; i < n.length; i++) {
    if (i === Math.floor(n.length / 2)) {
      continue;
    } else if (n[i] < pivot) {
      left.push(n[i]);
    } else {
      right.push(n[i]);
    }
  }

  return [...QuickSort(left), pivot, ...QuickSort(right)];
}
const res = QuickSort(arr);
console.log('quick sort -> ',res)

// Insertion Sort
function InsertionSort(n) {
  for (let i = 1; i < n.length; i++) {
    let current = n[i];
    let j = i - 1;

    while (j >= 0 && n[j] > current) {
      n[j + 1] = n[j];
      j--;
    }
    n[j + 1] = current;
  }
  console.log('insertion sort -> ',n);
}
InsertionSort(arr);

//Merge Sort
function MergeSort(n) {
  if (n.length <= 1) {
    return n;
  }

  let mid = Math.floor(n.length / 2);
  let left = n.slice(0, mid);
  let right = n.slice(mid);

  return merge(MergeSort(left), MergeSort(right));
}
const result = MergeSort(arr);
console.log('merge sort -> ',result)

function merge(left, right) {
  let res = [];
  let left_index = 0;
  let right_index = 0;

  while (left_index < left.length && right_index < right.length) {
    if (left[left_index] < right[right_index]) {
      res.push(left[left_index]);
      left_index++;
    } else {
      res.push(right[right_index]);
      right_index++;
    }
  }
  return res.concat(left.slice(left_index)).concat(right.slice(right_index));
}
