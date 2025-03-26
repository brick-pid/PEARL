problem = """
\"\"\"
Return the factorial of n.
>>> factorial(2)  # 2
>>> factorial(0)  # 1
\"\"\"
function factorial(n::Int)
"""

cot = """
1. **Understanding Factorial**: Recognize that the factorial of a number `n` is the product of all positive integers from 1 to `n`. 
2. **Choosing the Approach**: Decide between iterative and recursive approaches. Opt for the iterative approach to avoid recursion limit issues for larger numbers. 
3. **Implementing the Iterative Approach**: Start with initializing a result variable to 1. Then, multiply it sequentially with every integer from 1 to `n`. 
4. **Handling Edge Case**: Account for the edge case where `n` is 0. By definition, 0! (0 factorial) equals 1. 
"""

result = """
\"\"\"
Return the factorial of n.
>>> factorial(2)  # 2
>>> factorial(0)  # 1
\"\"\"
function factorial(n::Int)
    if n == 0
        return 1
    end
    result = 1
    for i in 1:n
        result *= i
    end
    return result
end
"""

knowledge = """
"""

relevant = """
\"\"\"
:type n: int
:rtype: int\"\"\"
function factorial(n)
    f = 1
    for i in 1:n
        f *= i
    end
    f
end

\"\"\"
Returns n! = n * (n-1) * (n-2) ... * 1
0! is 1.  Factorial is undefined for integers < 0.
Examples:
    factorial(0) returns 1
    factorial(2) returns 2
    factorial(3) returns 6
    factorial(5) returns 120
Parameter n: The integer for the factorial
Precondition: n is an int >= 0\"\"\"
function factorial(n)
    if n >= 0
        result = 1
        for i in 1:n
            result *= i
        end
        return result
    else
        error("n must be a non-negative integer.")
    end
end

\"\"\"
Computes n!
\"\"\"
function factorial(n::Int64)::Int64
    # It's OK to use a Int64 here, because you'll use the same one in the end anyway
    # You could also use a BigInt, but this would be a little slower.
    result::Int64 = 1
    # Loop from 1 to n
    for i = 1:n
        # Multiply result by i
        result *= i
    end
    return result
end

\"\"\"
Finding factorial iteratively
\"\"\"
function fact(n)
    prod = 1
    for i in 1:n
        prod *= i
    end
    return prod
end

\"\"\"
Iterative implementation of factorial algorithm.
factorial(0) = 1
factorial(1) = 1
:param n: positive integer
\"\"\"
function factorial(n::Int64)::Int64
    # if n == 0:
    #     return 1
    result = 1
    while n > 0
        result *= n
        n -= 1
    end
    return result
end
"""