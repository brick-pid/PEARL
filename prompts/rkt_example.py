"""
1. problem;
2. cot;
3. relevant;
4. result;
"""


problem = """
#lang racket

;; Return the factorial of n. 
;; (factorial 2) 
;; 2
;; (factorial 0) 
;; 1 
(define (factorial n)
"""

# cot = """
# 1. **Understanding Factorial**: Recognize that the factorial of a non-negative integer `n` (denoted as `n!`) is defined as the product of all positive integers from 1 to `n`. Specifically, \( n! = n \times (n - 1) \times (n - 2) \times \ldots \times 1 \). Additionally, it is important to note that by convention, \( 0! = 1 \).
# 2. **Choosing the Approach**: Consider the advantages and disadvantages of iterative versus recursive methods. The iterative approach is chosen to avoid potential recursion limit issues and stack overflow errors, especially when dealing with larger numbers. This ensures a more efficient use of memory and processing.
# 3. **Implementing the Iterative Approach**:
#    3.1. **Initialize a Result Variable**: Start by initializing a result variable (let’s call it `result`) to 1, as multiplying by 1 will not affect the outcome.
#    3.2. **Sequential Multiplication**: Use a loop (e.g., a `for` loop) to iterate through all integers from 1 to `n`, multiplying `result` by each integer in the sequence. This will accumulate the factorial product.
# 4. **Handling Edge Cases**: Specifically address edge cases:
#    4.1. If `n` is 0, return 1 immediately, as \( 0! = 1 \) by definition. 
#    4.2. Optionally, handle negative inputs by returning an error message or a specific value, since factorials are only defined for non-negative integers.
# 5. **Finalizing the Implementation**: Ensure the code is clear and includes comments for readability. Validate the implementation with various test cases, including small integers, larger integers, and edge cases like `0` and negative numbers, to confirm correctness and robustness.
# """

cot= """
1. Understand the problem: We need to write a function `factorial` that calculates the factorial of a given number `n`. Factorial of a number `n` is defined as the product of all positive integers from 1 to `n` (or equivalently, 1 × 2 × 3 × ... × `n`). By definition:

- Factorial of 0 is 1 (i.e., `0! = 1`).
- Factorial of any positive integer `n` is `n * (n-1) * (n-2) * ... * 1`.

For example:
- `factorial(0)` should return `1`.
- `factorial(2)` should return `2 * 1 = 2`.
- `factorial(3)` should return `3 * 2 * 1 = 6`.

2. Choose the approach to solve the problem (suitable for functional programming, e.g. Racket):
A common functional approach to solving such problems involves recursion. In recursion, a function calls itself, breaking the problem into simpler subproblems until it reaches a base case (the simplest form of the problem, which we know how to solve directly).

For factorial:
- The base case is when `n = 0`, for which the result is `1`.
- The recursive case is when `n > 0`. The result for `factorial(n)` can be expressed as `n * factorial(n - 1)`.

3. Break the problem into base case and recursive case:
- Base case: If `n` is 0, return 1.
- Recursive case: If `n` is greater than 0, return `n * factorial(n - 1)`.
"""

result = """
#lang racket
(define (factorial n)
  (if (= n 0)           ;; Base case: when n is 0, return 1
      1
      (* n (factorial (- n 1)))))  ;; Recursive case: n * factorial(n - 1)
"""

# result = """
# #lang racket

# (define (factorial n)
#   ; Handle the base case: 0! = 1
#   (if (= n 0)
#       1
#       ; For n > 0, calculate factorial iteratively
#       (let ([result 1])
#         ; Iterate from 1 to n, inclusive
#         (for ([i (in-range 1 (+ n 1))])
#           ; Multiply the current result by i
#           (set! result (* result i)))
#         ; Return the final result
#         result)))
# """

knowledge = """
Knowledge: In Racket, recursion is often preferred over iteration due to its efficient handling of tail recursion and lack of stack overflow.
Demonstrating Code
# racket
(define (remove-dups l) 
  (cond ([empty? l] empty)
        ([empty? (rest l)] l)
        [else (let ([i (first l)]) 
                (if (equal? i (first (rest l))) 
                    (remove-dups (rest l)) 
                    (cons i (remove-dups (rest l)))))]))

Knowledge: The `in-range` function generates sequences of numbers, providing flexibility with optional starting numbers, ending limits, and step sizes.
Demonstrating Code
# racket
(for (i ([in-range 3)]) (display i))

Knowledge: The `set!` expression evaluates `expr` and assigns its resulting value to `id`, which must be bound in the enclosing environment. The result of the `set!` expression is `#<void>`.
Demonstrating Code
# racket
(set! id expr)

Knowledge: Function calls in Racket are made using the syntax `(‹id› ‹expr›*)`, where the number of expressions corresponds to the number of arguments supplied to the function.
Demonstrating Code
# racket
(string-append "rope" "twine" "yarn")

Knowledge: Using specific fast-clause forms in for iterations can significantly improve performance, making them as efficient as hand-written loops tailored for specific data types.
Demonstrating Code
# racket
(time (for (i ([in-range 100000]) (for ([elem '(a b c d e f g h)]) (void))))

"""

relevant = """
#lang racket
;; Factorial
(define (factorial n)
        (if (< n 2)
                1
                (* n (factorial (- n 1)))))

#lang racket
;; Defined my own factorial just in case using python2.5 or less.
;; :param n:
;; :return:
(define (factorial n)
        (if (> n 1)
                (* n (factorial (- n 1)))
                1))

#lang racket
;; returns the factorial of num using a recursive method.
(define (factorial_recursive num)
        (cond 
                ((= num 0) 1) 
                (else (* num (factorial_recursive (- num 1))))))

#lang racket
;; This function calculate factorial using recursion.
(define (factorial_recursion integer)
        (if (<= integer 1)
                1
                (* integer (factorial_recursion (- integer 1)))
        )
)

#lang racket
;; Compute the factorial of natural number n.
(define (factorial n)
        (cond 
                ((= n 1) 1) 
                ((= n 0) 1)
                (else (* n (factorial (- n 1))))))
"""