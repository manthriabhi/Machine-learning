import random

# 1. Count vowels and consonants
def count_vowels_consonants(text):
    vowels = "aeiouAEIOU"
    v = c = 0
    for ch in text:
        if ch.isalpha():
            if ch in vowels:
                v += 1
            else:
                c += 1
    return v, c


# 2. Matrix multiplication
def multiply_matrices(A, B):
    if len(A[0]) != len(B):
        return None

    result = [[0]*len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


# 3. Count common elements between two lists
def count_common_elements(list1, list2):
    return len(set(list1) & set(list2))


# 4. Transpose of a matrix
def transpose_matrix(matrix):
    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]


# 5. Mean, Median, Mode
def mean(numbers):
    return sum(numbers) / len(numbers)

def median(numbers):
    numbers = sorted(numbers)
    mid = len(numbers) // 2
    return (numbers[mid-1] + numbers[mid]) / 2

def mode(numbers):
    freq = {}
    for n in numbers:
        freq[n] = freq.get(n, 0) + 1
    max_freq = max(freq.values())
    return [n for n in freq if freq[n] == max_freq]

# Program 1
text = input("Enter a string: ")
vowels, consonants = count_vowels_consonants(text)
print("Vowels:", vowels, "Consonants:", consonants)

# Program 2
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]
product = multiply_matrices(A, B)
print("Matrix Product:" if product else "Matrices cannot be multiplied")
if product:
    for row in product:
        print(row)

# Program 3
list1 = [1, 2, 3, 4]
list2 = [3, 4, 5, 6]
print("Common elements count:", count_common_elements(list1, list2))

# Program 4
matrix = [[1, 2, 3], [4, 5, 6]]
print("Transpose:")
for row in transpose_matrix(matrix):
    print(row)

# Program 5
numbers = [random.randint(100, 150) for _ in range(100)]
print("Mean:", mean(numbers))
print("Median:", median(numbers))
print("Mode:", mode(numbers))
