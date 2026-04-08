/* 
 * Author: Isaac H. Lopez Diaz
 * Description: Core Types
 * Licensed under GNU GPLv3
*/

#include "../include/core.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

Matrix* init_matrix(size_t element_size, int rows, int cols) {
    Matrix* mtx = malloc(sizeof(Matrix));

    // Handle errors for allocating memory
    if (!mtx) return NULL;

    mtx->data = calloc(rows * cols, element_size);
    if (!mtx->data) {
        free(mtx);
        return NULL;
    }

    mtx->strides = malloc(2 * sizeof(int));
    if (!mtx->strides) {
        free(mtx->data);
        free(mtx);
        return NULL;
    }
    mtx->strides[0] = cols*element_size;
    mtx->strides[1] = element_size;

    mtx->rows = rows;
    mtx->cols = cols;
    mtx->capacity = rows * cols;
    mtx->element_size = element_size;

    return mtx;
}

Matrix* transpose(Matrix* mtx) {
    Matrix* mtxT = init_matrix(mtx->element_size, mtx->cols, mtx->rows);

    if (!mtxT) return NULL;

    int i;
    for (i = 0; i < mtx->rows; i++) {
        int k;
        for (k = 0; k < mtx->cols; k++) {
            void* src = (char*) mtx->data + i * mtx->strides[0] + k * mtx->strides[1];
            void* dst = (char*) mtxT->data + k * mtxT->strides[0] + i * mtxT->strides[1];
            memcpy(dst, src, mtx->element_size);
        }
    }

    return mtxT;
}

Matrix* sum_matrix(Matrix* a, Matrix* b);
Matrix* mul_matrix(Matrix* a, Matrix* b);

void free_matrix(Matrix* mtx) {
    if (!mtx) return;

    free(mtx->data);
    free(mtx->strides);
    free(mtx);
}
