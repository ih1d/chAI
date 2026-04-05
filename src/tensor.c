#include "../include/tensor.h"
#include <string.h>

static size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_FLOAT64: return sizeof(double);
        case DTYPE_INT32:   return sizeof(int);
    }
    return 0;
}

// ============================================================
// Lifecycle
// ============================================================

Tensor* create(int* shape, int dim, DType dtype, void* data) {
    if (!shape || dim <= 0) return NULL;

    Tensor* t = malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->shape = malloc(dim * sizeof(int));
    if (!t->shape) { free(t); return NULL; }
    memcpy(t->shape, shape, dim * sizeof(int));

    t->dim = dim;
    t->dtype = dtype;

    t->strides = malloc(dim * sizeof(int));
    if (!t->strides) { free(t->shape); free(t); return NULL; }

    t->strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * shape[i + 1];
    }

    t->size = 1;
    for (int i = 0; i < dim; i++) {
        t->size *= shape[i];
    }

    size_t bytes = t->size * dtype_size(dtype);

    if (data) {
        t->data = malloc(bytes);
        if (!t->data) { free(t->strides); free(t->shape); free(t); return NULL; }
        memcpy(t->data, data, bytes);
    } else {
        t->data = calloc(t->size, dtype_size(dtype));
        if (!t->data) { free(t->strides); free(t->shape); free(t); return NULL; }
    }

    return t;
}

Tensor* zeros(int* shape, int dim, DType dtype) {
    return create(shape, dim, dtype, NULL);
}

Tensor* ones(int* shape, int dim, DType dtype) {
    Tensor* t = create(shape, dim, dtype, NULL);
    if (!t) return NULL;

    for (size_t i = 0; i < t->size; i++) {
        switch (dtype) {
            case DTYPE_FLOAT32: ((float*)t->data)[i]  = 1.0f; break;
            case DTYPE_FLOAT64: ((double*)t->data)[i]  = 1.0;  break;
            case DTYPE_INT32:   ((int*)t->data)[i]     = 1;    break;
        }
    }

    return t;
}
Tensor* full(int* shape, int dim, DType dtype, double value);
Tensor* tensor_rand(int* shape, int dim, DType dtype);
Tensor* randn(int* shape, int dim, DType dtype);
Tensor* eye(int n, DType dtype);
Tensor* clone(const Tensor* t);
void    free_tensor(Tensor* t);

// ============================================================
// Type queries
// ============================================================

int is_scalar(const Tensor* t);
int is_vector(const Tensor* t);
int is_matrix(const Tensor* t);

// ============================================================
// Element access
// ============================================================

double get(const Tensor* t, int* indices);
void   set(Tensor* t, int* indices, double value);
void   fill(Tensor* t, double value);

// ============================================================
// Shape manipulation
// ============================================================

Tensor* reshape(const Tensor* t, int* new_shape, int new_dim);
Tensor* transpose(const Tensor* t);
Tensor* permute(const Tensor* t, int* axes);
Tensor* flatten(const Tensor* t);
Tensor* squeeze(const Tensor* t, int axis);
Tensor* unsqueeze(const Tensor* t, int axis);
Tensor* slice(const Tensor* t, int axis, int start, int end);
Tensor* concat(const Tensor* a, const Tensor* b, int axis);

// ============================================================
// Elementwise arithmetic
// ============================================================

Tensor* add(const Tensor* a, const Tensor* b);
Tensor* sub(const Tensor* a, const Tensor* b);
Tensor* mul(const Tensor* a, const Tensor* b);
Tensor* tensor_div(const Tensor* a, const Tensor* b);
Tensor* neg(const Tensor* t);
Tensor* scale(const Tensor* t, double scalar);
Tensor* pow(const Tensor* t, double exponent);
Tensor* sqrt(const Tensor* t);
Tensor* tensor_abs(const Tensor* t);

// ============================================================
// Matrix / linear algebra
// ============================================================

Tensor* matmul(const Tensor* a, const Tensor* b);
Tensor* dot(const Tensor* a, const Tensor* b);
double  norm(const Tensor* t);

// ============================================================
// Reduction operations
// ============================================================

Tensor* sum(const Tensor* t, int axis);
Tensor* mean(const Tensor* t, int axis);
Tensor* max(const Tensor* t, int axis);
Tensor* min(const Tensor* t, int axis);
Tensor* argmax(const Tensor* t, int axis);
Tensor* argmin(const Tensor* t, int axis);

// ============================================================
// Activation functions (deep learning essentials)
// ============================================================

Tensor* exp(const Tensor* t);
Tensor* log(const Tensor* t);
Tensor* relu(const Tensor* t);
Tensor* sigmoid(const Tensor* t);
Tensor* tanh_act(const Tensor* t);
Tensor* softmax(const Tensor* t, int axis);

// ============================================================
// Comparison (return tensors of 0/1)
// ============================================================

Tensor* eq(const Tensor* a, const Tensor* b);
Tensor* gt(const Tensor* a, const Tensor* b);
Tensor* lt(const Tensor* a, const Tensor* b);
Tensor* clamp(const Tensor* t, double min_val, double max_val);

// ============================================================
// Utility
// ============================================================

void print(const Tensor* t);
int  equal(const Tensor* a, const Tensor* b);

