problem = """
-- Return the factorial of n.
-- print(factorial(2))  --> 2
-- print(factorial(0))  --> 1
function factorial(n)
"""

cot = """
1. **Understanding Factorial**: Recognize that the factorial of a number `n` is the product of all positive integers from 1 to `n`. 
2. **Choosing the Approach**: Decide between iterative and recursive approaches. Opt for the iterative approach to avoid recursion limit issues for larger numbers. 
3. **Implementing the Iterative Approach**: Start with initializing a result variable to 1. Then, multiply it sequentially with every integer from 1 to `n`. 
4. **Handling Edge Case**: Account for the edge case where `n` is 0. By definition, 0! (0 factorial) equals 1. 
"""

result = """
-- Return the factorial of n.
-- print(factorial(2))  --> 2
-- print(factorial(0))  --> 1
function factorial(n)
    -- Handle the edge case where n is 0
    if n == 0 then
        return 1
    end
    
    -- Initialize result variable
    local result = 1
    
    -- Iterate from 1 to n and multiply the result
    for i = 1, n do
        result = result * i
    end
    
    return result
end
"""

knowledge = ""

relevant = """
-- Return n! (0! is 1).
local function factorial(n)
        local function iter(f, n)
                if n <= 1 then return 1 end
                return f(f, n - 1) * n
        end
        return iter(iter, n)
end

-- Returns the factorial of the positive integer n using
-- loops.
local function factorialFinderLoops(n)
    if type(n) ~= 'number' then
        error('n should be a number.')
        return nil
    elseif n < 0 then
        error('n should be a positive integer.')
        return nil
    end
    local result = 1
    for i = 2, n do
        result = result * i
    end
    return result
end

-- This function returns the factorial of n (denoted n!)
-- Input: n (number to compute the factorial of)
-- Returns: value of n factorial
-- Doctests:
-- >>> factorial(3)
-- 6
-- >>> factorial(1)
-- 1
-- >>> factorial(0)
-- 1
local function factorial(n)
    --[[
    * 1. If the input value is 0 or 1, return the value.
    ]]--
    if n == 0 or n == 1 then
        return 1
    end
    --[[
    * 2. If the input value is 0 or 1, return the value.
    * 3. Otherwise, recursively compute the factorial of n - 1
    ]]--
    return n * factorial(n - 1)
end

-- Return the factorial of n.
-- Parameters
-- ----------
-- n :
--     an integer of which the factorial is evaluated.
-- Returns
-- -------
-- result :
--     The factorial of n.
local function factorial(n)
  local result = 1
  for x = 2, n do
    result = result * x
  end
  return result
end

-- Calculates factorial of a given number
-- Uses Recursive approach
-- :param num: the number to calculate the factorial
-- :return: the factorial result
local function factorial_recursive(num)
  if num == 0 then
    return 1
  end
  return num * factorial_recursive(num - 1)
end

"""