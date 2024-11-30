function max(a: nat, b: nat) : nat
{
  if a > b
  then a
  else b
}

function min(a: nat, b: nat) : nat
{
  if a < b
  then a
  else b
}


function gcd(a: nat, b: nat): nat
  decreases b
{
  if b == 0
  then a
  else gcd(b, a % b)
}

predicate Divides(d: nat, n: nat)
  decreases n
  requires d > 0
{
  n % d == 0
}

method EuclidGCD(a: nat, b: nat) returns (resultGcd: nat)
  requires a > 0 && b > 0
  ensures resultGcd > 0
  ensures Divides(resultGcd, a) && Divides(resultGcd, b)
  ensures forall g: nat :: g > 0 && Divides(g, a) && Divides(g, b) ==> g <= resultGcd
{

  var x: nat := a;
  var y: nat := b;


  while y != 0
    invariant x > 0 || y > 0
    invariant gcd(x, y) == gcd(a, b)
  {
    var temp: nat := y;
    y := x % y;
    x := temp;
  }

  resultGcd := x;

}

// Exercice 1
// method Main() {
//   var r := EuclidGCD(12, 8);

//   print "Result ", r;
// }

// Exercice 2
// this do sum of squares of first n natural numbers
// https://math.stackexchange.com/q/48080
method m4(n: int) returns (s: int)
  requires n >= 0
  ensures s == (n * (n + 1) * (2 * n + 1)) / 6
{
  var i,k : int;
  s := 0;
  k := 1;
  i := 1;
  while (i <= n)
    invariant 0 <= i <= n + 1
    invariant s == (i - 1) * i * (2 * (i - 1) + 1) / 6
    invariant k == i * i
  {
    s := s + k;
    k := k + 2 * i + 1;
    i := i + 1;
  }
}

method Main() {
  var r := m4(3);

  print "Result ", r;
}

// Exercice 3

method TwoLargestElements(arr: array<int>) returns (max1: int, max2: int)
  requires arr.Length >= 2
  ensures forall i :: 0 <= i < arr.Length ==> arr[i] <= max1
  ensures forall i :: 0 <= i < arr.Length ==> arr[i] <= max2 || max1 == max2
  ensures exists i, j :: 0 <= i < arr.Length && 0 <= j < arr.Length && i != j && arr[i] == max1 && arr[j] == max2
{

  var maxIdx1 := 0;
  var maxIdx2 := 1;
  var i: int := 2;

  if arr[0] < arr[1] {
    maxIdx1, maxIdx2 := 1, 0;
  }

  while i < arr.Length - 1
    invariant 2 <= i <= arr.Length
    invariant forall j :: 0 <= j < i ==> arr[j] <= arr[maxIdx1]
    invariant forall j :: 0 <= j < i ==> arr[j] <= arr[maxIdx2] || maxIdx1 == maxIdx2
    invariant exists j, k :: 0 <= j < i && 0 <= k < i && j != k && arr[j] == arr[maxIdx1] && arr[k] == arr[maxIdx2]
  {
    if arr[i] > arr[maxIdx1] {
      maxIdx2 := maxIdx1;
      maxIdx1 := i;
    } else if arr[i] > arr[maxIdx2] {
      maxIdx2 := i;
    }
    i := i + 1;
  }

  max1 := arr[maxIdx1];
  max2 := arr[maxIdx2];
}
