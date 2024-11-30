function add(x : int, y : int) : int
{
  x + y
}

// method Main() {
//     var y := add(2,4);
//     print "Sum is", y, "\n";
// }

method Add(x : int, y : int) returns (s : int)
  ensures s == x + y
{
  s := x + y;
}

// method Main() {
//     var s := Add(2,4);
//     print "Sum is ", s, "\n";
// }

// {P} S {Q}
//  (y >= 4)[x + 1/y]   y := x + 1 {y >= 4}
//  (x + 1 >= 4)   y := x + 1 {y >= 4}
method inc(x : int) returns (y : int)
{
  assume(x + 1 >= 4);
  y := x + 1;
  assert(y >= 4);
}

method inc'(x : int) returns (y : int)
  requires x >= 3
  ensures y >= 4
{
  y := Add(x, 1);
}

method Abs(x : int) returns (m : int)
  ensures x < 0 ==> m == -x
  ensures x >= 0 ==> m == x
  ensures m >= 0
  ensures m >= x
{
  if (x < 0) {
    m := -x;
  } else {
    m := x;
  }
}

method increment(n : int) returns (i : int)
  requires n >= 0
  ensures i == n
{
  i := 0;
  while i < n
    invariant i <= n
  {
    i := i + 1;
  }
  // i <= n /\ !(i < n) <-> i <= n /\ i >= n <-> i == n
}

method mysum(n : int) returns (s : int)
  requires n >= 0
  ensures s == n * (n + 1) / 2
{
  var i := 1;
  s := 0;
  while i <= n
    invariant s == i * (i - 1) / 2
    invariant i <= n + 1
  {
    s := s + i;
    i := i + 1;
  }
}

method Mod2(n : int) returns (r : int)
  requires n >= 0
  ensures r == n % 2
{
  r := n;
  while r >= 2
    invariant r >= 0
    invariant r % 2 == n % 2
  {
    r := r - 2;
  }
}

function factSpec(n : int) : int {
  if n <= 0
  then 1
  else n * factSpec(n - 1)
}

method fact(n : int) returns (r : int)
  requires n >= 0
  ensures r == factSpec(n)
{
  var k := 1;
  r := 1;
  while k <= n
    invariant r == factSpec(k - 1)
    invariant k <= n + 1
  {
    r := r * k;
    k := k + 1;
  }
}

predicate maxArr(a : seq<int>, m : int) {
  exists j :: 0 <= j < |a| && a[j] == m && forall i :: 0 <= i < |a| ==> a[i] <= m
}

method max(a : array<int>) returns (r : int)
  requires a.Length > 0
  ensures maxArr(a[..], r)
{
  r := a[0];
  for i := 1 to a.Length
    invariant maxArr(a[..i],r)
  {
    if r < a[i] {
      r := a[i];
    }
  }
}

method div(x : int, y : int) returns (q :int, r:int)
  requires y > 0
  ensures x == y * q + r
  //    decreases *
{
  q := 0;
  r := x;
  while r >= y
    invariant x == y * q + r
    // decreases x - y * q
    // decreases r - y
  {
    q := q + 1;
    r := r - y;
  }
}

method Main() {
  var y, z := div(25,4);
  print "q = ", y, "r = ", z, "\n";
}