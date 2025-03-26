problem = """
# Return the factorial of n.
# >>> factorial(2)  # 2
# >>> factorial(0)  # 1

factorial <- function(n)
"""

cot = """
1. **Understanding Factorial**: Recognize that the factorial of a number `n` is the product of all positive integers from 1 to `n`. 
2. **Choosing the Approach**: Decide between iterative and recursive approaches. Opt for the iterative approach to avoid recursion limit issues for larger numbers. 
3. **Implementing the Iterative Approach**: Start with initializing a result variable to 1. Then, multiply it sequentially with every integer from 1 to `n`. 
4. **Handling Edge Case**: Account for the edge case where `n` is 0. By definition, 0! (0 factorial) equals 1. 
"""

result = """
# Return the factorial of n.
# >>> factorial(2)  # 2
# >>> factorial(0)  # 1

factorial <- function(n) {
  if (n == 0) {
    return(1)
  }
  
  result <- 1
  for (i in 1:n) {
    result <- result * i
  }
  
  return(result)
}
"""

knowledge = ""

relevant = """
factorial <- function(n) {
    result <- 1
    for (i in 1:n) {
        result <- result * i
    }
    return(result)
}

# >>> my_factorial1(1)
# 1
# >>> my_factorial1(0)
# 1
# >>> my_factorial1(-1)
# 1
# >>> my_factorial1(5)
# 120
my_factorial1 <- function(n) {
  if (n < 2) {
    return(1)
  } else {
    return(n * my_factorial1(n - 1))
  }
}

# Compute n! where n is an integer >= 0.
factorial <- function(n) {
    if (n == 0) {
        return(1)
    } else {
        return(n * factorial(n - 1))
    }
}

# Factorial value for positive integer n.
factorial <- function(n) {
    if (n == 0) {
        return(1)
    } else {
        return(n * factorial(n - 1))
    }
}


# return n factorial    
# The factorial is defined as n! = n*(n-1)!
# Note: The recursive factoial implementation will break down if n is too large
factorial <- function(n) {
    if (n == 0) return(1)
    else return (n*factorial(n-1))
}
"""