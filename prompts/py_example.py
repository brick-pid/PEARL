problem = """
def factorial(n): 
    \"\"\" Return the factorial of n. 
    >>> factorial(2) 
    2
    >>> factorial(0) 
    1 
    \"\"\" 
"""

cot = """
1. **Understanding Factorial**: Recognize that the factorial of a number `n` is the product of all positive integers from 1 to `n`. 
2. **Choosing the Approach**: Decide between iterative and recursive approaches. Opt for the iterative approach to avoid recursion limit issues for larger numbers. 
3. **Implementing the Iterative Approach**: Start with initializing a result variable to 1. Then, multiply it sequentially with every integer from 1 to `n`. 
4. **Handling Edge Case**: Account for the edge case where `n` is 0. By definition, 0! (0 factorial) equals 1. 
"""

result = """
def factorial(n): 
    \"\"\" Return the factorial of n. 
    >>> factorial(2) 
    2
    >>> factorial(0) 
    1 
    \"\"\" 
    if n == 0: 
        return 1 
    result = 1 
    for i in range(1, n + 1): 
        result *= i 
    return result 
"""