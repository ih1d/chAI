#include "../include/core.h"
#include <stdlib.h>
#include <stdio.h>

Matrix* init_matrix(size_t element_size, int rows, int cols) {
    Matrix* mtx = malloc(sizeof(Matrix));

    if (!mtx) return NULL;

    mtx->strides = malloc(sizeof(int)*2);
    if (!mtx->strides) {
        free(mtx);
        return NULL;
    }

    mtx->data = malloc(element_size * rows * cols);
    if (!mtx->data) {
        free(mtx->strides);
        free(mtx);
        return NULL;
    }

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
    int k;
    for (i = 0; i < mtx->rows; i++) {
        for (k = 0; k < mtx->cols; k++) {

        }
    }

    return mtxT;
}

void free_matrix(Matrix* mtx) {
    if (!mtx) return;
    free(mtx->strides);
    free(mtx->data);
    free(mtx);
}
