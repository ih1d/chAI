#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/core.h"

int main() {

    Matrix* mtx = init_matrix(sizeof(int), 2, 3);
    int i;
    int k;

    for (i = 0; i < mtx->rows; i++)
        for (k = 0; k < mtx->cols; k++)
            MATRIX_AT(mtx, int, i, k) = i * 2 + k;

    printf("Matrix A:\n");
    for (i = 0; i < mtx->rows; i++) {
        for (k = 0; k < mtx->cols; k++) {
            if(k == mtx->cols - 1) {
                int src = MATRIX_AT(mtx, int, i, k);
                printf("%d %d: %d", i+1, k+1, src);
            }
            else {
                int src = MATRIX_AT(mtx, int, i, k);
                printf("%d %d: %d, ", i+1, k+1, src);
            }
        }
        printf("\n");
    }
   

    printf("Matrix AT:\n");
    Matrix* mtxT = transpose(mtx);
    for (i = 0; i < mtxT->rows; i++) {
        for (k = 0; k < mtxT->cols; k++) {
            if(k == mtxT->cols - 1) {
                int src = MATRIX_AT(mtxT, int, i, k);
                printf("%d %d: %d", i+1, k+1, src);
            }
            else {
                int src = MATRIX_AT(mtxT, int, i, k);
                printf("%d %d: %d, ", i+1, k+1, src);
            }
        }
        printf("\n");
    }

    free_matrix(mtx);
}
