import numpy as np

# Skalar
# scalar = 3.3
# print(scalar)
# print(type(scalar))

# Wektor (tensor 1 rzędu)
vector = np.array([1,2,3.2,-4,5,-2,3])
print(vector)
print(type(vector))
print(f'Rozmiar wektora: {vector.shape}')
print(f'Typ danych wektora: {vector.dtype}')
print(f'Rząd: {vector.ndim}')
print(f'Długość: {len(vector)}')

# Macierz (tensor 2 rzędu)
array = np.array([[2, 6, 3],
                [5, -3, 4]], dtype="float")
print(array)
print(type(array))
print(f'Rozmiar macierzy: {array.shape}')
print(f'Typ danych macierzy: {array.dtype}')
print(f'Rząd: {array.ndim}')
print(f'Długosc: {len(array)}')

# Tensor (3 rzędu)
tensor = np.array([
    [[1,2,3],
     [4,5,6]],
    [[7,8,9],
     [0,1,2]]
])

print(tensor)
print(type(tensor))
print(f'Rozmiar macierzy: {tensor.shape}')
print(f'Typ danych macierzy: {tensor.dtype}')
print(f'Rząd: {tensor.ndim}')
print(f'Długosc: {len(tensor)}')

