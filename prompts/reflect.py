relevance_prompt = """
You are a highly intelligent and reliable coding assistant, dedicated to delivering accurate and trustworthy information. 
You will be given a problem, a pseudocode implementation, and a knowledge. You should reflect on the knowledge to decide whether this knowledge is relevant to implement the pseudocode.
You should first give rationale of this knowledge is relevant or not, and then give the final decision.

### Problem:
#lang racket

;; Return the factorial of n. 
;; (factorial 2) 
;; 2
;; (factorial 0) 
;; 1 
(define (factorial n)

### Pseudocode Implementation:
1. **Understanding Factorial**: Recognize that the factorial of a non-negative integer `n` (denoted as `n!`) is defined as the product of all positive integers from 1 to `n`. Specifically, \( n! = n \times (n - 1) \times (n - 2) \times \ldots \times 1 \). Additionally, it is important to note that by convention, \( 0! = 1 \).
2. **Choosing the Approach**: Consider the advantages and disadvantages of iterative versus recursive methods. The iterative approach is chosen to avoid potential recursion limit issues and stack overflow errors, especially when dealing with larger numbers. This ensures a more efficient use of memory and processing.
3. **Implementing the Iterative Approach**:
   3.1. **Initialize a Result Variable**: Start by initializing a result variable (let’s call it `result`) to 1, as multiplying by 1 will not affect the outcome.
   3.2. **Sequential Multiplication**: Use a loop (e.g., a `for` loop) to iterate through all integers from 1 to `n`, multiplying `result` by each integer in the sequence. This will accumulate the factorial product.
4. **Handling Edge Cases**: Specifically address edge cases:
   4.1. If `n` is 0, return 1 immediately, as \( 0! = 1 \) by definition. 
   4.2. Optionally, handle negative inputs by returning an error message or a specific value, since factorials are only defined for non-negative integers.
5. **Finalizing the Implementation**: Ensure the code is clear and includes comments for readability. Validate the implementation with various test cases, including small integers, larger integers, and edge cases like `0` and negative numbers, to confirm correctness and robustness.



"""