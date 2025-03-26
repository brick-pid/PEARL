problem = """
(* 
   Return the factorial of n.
   Example:
   factorial 2;;  (* 2 *)
   factorial 0;;  (* 1 *)
*)
let rec factorial n = 
"""

cot = """
1. **Understanding Factorial**: Recognize that the factorial of a number `n` is the product of all positive integers from 1 to `n`. 
2. **Choosing the Approach**: Decide between iterative and recursive approaches. Opt for the iterative approach to avoid recursion limit issues for larger numbers. 
3. **Implementing the Iterative Approach**: Start with initializing a result variable to 1. Then, multiply it sequentially with every integer from 1 to `n`. 
4. **Handling Edge Case**: Account for the edge case where `n` is 0. By definition, 0! (0 factorial) equals 1. 
"""

result = """
(* 
   Return the factorial of n.
   Example:
   factorial 2;;  (* 2 *)
   factorial 0;;  (* 1 *)
*)
let rec factorial n = 
  if n = 0 then 1
  else n * factorial (n - 1)
"""

knowledge = ""

relevant = """
(**
 * This is a recursive function that calls
itself to find the factorial of given number
*)
let factorial (num : int) : int =
  let rec loop (n : int) (acc : int) : int =
    if n = 1 then acc
    else loop (n-1) (n*acc) in
  loop num 1

(**
 * Return n! for postive values of n.
>>> factorial(1)
1
>>> factorial(2)
2
>>> factorial(3)
6
>>> factorial(4)
24
*)
let factorial (n : int) : int =
  let rec fact acc n =
    if n = 0 then
      acc
    else
      fact (acc * n) (n - 1)
  in
  fact 1 n

(**
 * Return the factorial of a natural number.
>>> factorial(0)
1
>>> factorial(5)
120
*)
let factorial (n : int) : int =
    let rec factorial_rec (n : int) (acc : int) : int =
        match n with
        | 0 -> acc
        | 1 -> acc
        | _ -> factorial_rec (n - 1) (acc * n)
    in
    factorial_rec n 1

(**
 * Returns n! = n * (n-1) * (n-2) ... * 1
0! is 1.  Factorial is undefined for integers < 0.
Examples:
 * factorial(0) returns 1
 * factorial(2) returns 2
 * factorial(3) returns 6
 * factorial(5) returns 120
Parameter n: The integer for the factorial
Precondition: n is an int >= 0
*)
let factorial (n : int) : int =
    let result = ref 1 in
    for i = 1 to n do
        result := !result * i
    done;
    !result

(**
 * Returns n!
*)
let factorial (n : int) : int =
  let p = ref 1 in
  for j = 1 to n do
    p := !p * j
  done;
  !p

"""
