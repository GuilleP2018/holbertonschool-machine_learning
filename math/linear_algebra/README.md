# Linear algebra README

# Learning Objectives


1. **Vector**: A vector is a one-dimensional array of numbers. For example, `v = [1, 2, 3]` in Python.

2. **Matrix**: A matrix is a two-dimensional array of numbers. For example, `m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]` in Python.

3. **Transpose**: The transpose of a matrix is a new matrix whose rows are the columns of the original. (This makes the first row of the original matrix become the first column of the new one, the second row become the second column, and so on.)

4. **Shape of a Matrix**: The shape of a matrix is a tuple that indicates the number of rows and columns in the matrix. For example, the shape of the matrix `m` above is `(3, 3)`.

5. **Axis**: In a 2D matrix, axis 0 represents rows and axis 1 represents columns.

6. **Slice**: A slice is a subset of a matrix or vector. In Python, you can slice using the `:` operator. For example, `v[1:3]` would return a vector containing the 2nd and 3rd elements of `v`.

7. **Element-wise Operations**: These are operations that are performed on corresponding elements of vectors or matrices. For example, adding two matrices together adds the elements at each corresponding position.

8. **Concatenation of Vectors/Matrices**: This is the process of joining one or more vectors or matrices end-to-end. In Python, you can use `numpy.concatenate()` to concatenate arrays.

9. **Dot Product**: The dot product of two vectors is the sum of the products of their corresponding entries. For example, the dot product of `[1, 2, 3]` and `[4, 5, 6]` is `1*4 + 2*5 + 3*6`.

10. **Matrix Multiplication**: This is a binary operation that takes a pair of matrices, and produces another matrix. Elements of the product are calculated according to the rule of matrix multiplication.

11. **Numpy**: Numpy is a Python library that provides support for large, multi-dimensional arrays and matrices, along with a large collection of mathematical functions to operate on these arrays.

12. **Parallelization**: This is the process of dividing a program into parts that can be executed in parallel to make the program run faster. It's important because it can greatly speed up processing time, especially for tasks that involve large amounts of data.

13. **Broadcasting**: In Numpy, broadcasting allows mathematical operations to be performed between arrays of different shapes. For example, you can add a scalar (a single number) to a matrix (an array of numbers), and Numpy will "broadcast" the scalar to all elements in the matrix.


