import numpy as np

print("Enter list of numbers")
nums = [int(x) for x in input().split()]

# Function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result

print(softmax(nums))