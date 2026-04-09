/* 
 * Author: Isaac H. Lopez Diaz
 * Description: Core Types
*/
#ifndef CORE_H
#define CORE_H

#include <stdlib.h>

/* Matrix */
typedef struct {
    void* data;
    size_t element_size;
    int rows;
    int cols;
    int* strides;
    int capacity;
} Matrix;

Matrix* init_matrix(size_t element_size, int rows, int cols);

Matrix* transpose(Matrix* mtx);

void free_matrix(Matrix* mtx);

#endif
