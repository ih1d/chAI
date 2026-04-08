/* 
 * Author: Isaac H. Lopez Diaz
 * Description: Core Types
 * Licensed under GNU GPLv3
*/

#ifndef CORE_H
#define CORE_H

#include <stdlib.h>

#define MATRIX_AT(mtx, T, i, j) \
        (*(T*)((char*)(mtx)->data + (i) * (mtx)->strides[0] + (j) * (mtx)->strides[1]))

/* Vector */
typedef struct {
    void* data;
    int rows;
    int cols;
    int capacity;
    int* strides;
    size_t element_size;
} Matrix;

Matrix* init_matrix(size_t element_size, int rows, int cols);

Matrix* transpose(Matrix* mtx);

Matrix* sum_matrix(Matrix* a, Matrix* b);

Matrix* mul_matrix(Matrix* a, Matrix* b);

void free_matrix(Matrix* mtx);

#endif
