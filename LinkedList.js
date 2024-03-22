class Node {
  constructor(data) {
    (this.data = data), (this.next = null);
  }
}

const node1 = new Node(11);
const node2 = new Node(21);
const node3 = new Node(31);
const node4 = new Node(41);

node1.next = node2;
node2.next = node3;
node3.next = node4;

// check if an element present in a linkedlist (Iterative Approach):
function check(head, X) {
  let c = head;
  while (c) {
    if (c.data === X) {
      return true;
    }
    c = c.next;
  }
  return false;
}

if (check(node1, 42)) {
  console.log("yes");
} else {
  console.log("no");
}

// check if an element present in a linkedlist (Recursive Approach):
function Check(head, X) {
  if (head === null) return false;

  if (head.data === X) {
    return true;
  }
  return Check(head.next, X);
}

if (Check(node1, 11)) {
  console.log("Present");
} else {
  console.log("Absent");
}

//Find Length of a Linked List (Iterative)

function CheckLengthIterative(head) {
  let count = 0;
  let c = head;
  while (c) {
    if (c.data !== null) {
      count++;
    }
    c = c.next;
  }
  console.log(count);
}
CheckLengthIterative(node1)

//Find Length of a Linked List (Recursive)
function CheckLengthRecursive(head) {
  if (head === null) return 0;

  return 1 + CheckLengthRecursive(head.next);
}
console.log(CheckLengthRecursive(node1))

// Reverse a Linked List  (Iterative)
function ReverseLinkedList(head) {
  let prev = null;
  let current = head;
  let next = null;

  while (current !== null) {
    next = current.next;
    current.next = prev;
    prev = current;
    current = next;
  }
  return prev;
}
1
const res = ReverseLinkedList(node1);
printLinkedList(res);

function printLinkedList(head) {
  let c = head;
  let d = []
  while (c !== null) {
    d.push(c.data)
    c = c.next;
  }
  console.log(d)
}
