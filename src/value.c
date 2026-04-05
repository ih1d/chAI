#include "../include/value.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Value* create_value(double data) {
    Value* self = malloc(sizeof(Value));
    self->data = data;
    self->grad = 0.0;
    self->children = NULL;
    self->n_children = 0;
    self->op = OP_NONE;
    self->_backward = NULL;
    return self;
}

Value* init_value(Value* self, double data, struct Value** children, size_t children_size, Op op) {
    if (!self)
        self = malloc(sizeof(Value));
    self->data = data;
    self->grad = 0.0;
    self->children = children;
    self->n_children = children_size;
    self->op = op;
    self->_backward = NULL;
    return self;
}

/* Backward functions */
static void add_backward(Value* self) {
    self->children[0]->grad += self->grad;
    self->children[1]->grad += self->grad;
}

static void mul_backward(Value* self) {
    self->children[0]->grad += self->children[1]->data * self->grad;
    self->children[1]->grad += self->children[0]->data * self->grad;
}

static void pow_backward(Value* self) {
    double exp = self->children[1]->data;
    self->children[0]->grad += (exp * pow(self->children[0]->data, exp - 1)) * self->grad;
}

static void relu_backward(Value* self) {
    self->children[0]->grad += self->data > 0 ? self->grad : 0;
}

/* Operations */
Value* add(Value* self, Value* other) {
    struct Value** children = malloc(2*sizeof(Value*));
    children[0] = self;
    children[1] = other;
    Value* out = init_value(NULL, self->data + other->data, children, 2, OP_ADD);
    out->_backward = add_backward;
    return out;
}

Value* mul(Value* self, Value* other) {
    struct Value** children = malloc(2*sizeof(Value*));
    children[0] = self;
    children[1] = other;
    Value* out = init_value(NULL, self->data * other->data, children, 2, OP_MUL);
    out->_backward = mul_backward;
    return out;
}

Value* pow_val(Value* self, double exponent) {
    struct Value** children = malloc(2*sizeof(Value*));
    children[0] = self;
    children[1] = create_value(exponent);
    Value* out = init_value(NULL, pow(self->data, exponent), children, 2, OP_POW);
    out->_backward = pow_backward;
    return out;
}

Value* relu(Value* self) {
    struct Value** children = malloc(sizeof(Value*));
    children[0] = self;
    Value* out = self->data < 0
        ? init_value(NULL, 0.0, children, 1, OP_RELU)
        : init_value(NULL, self->data, children, 1, OP_RELU);
    out->_backward = relu_backward;
    return out;
}

Value* neg(Value* self) {
    Value* other = create_value(-1.0);
    return mul(self, other);
}

Value* sub(Value* self, Value* other) {
    return add(self, neg(other));
}

Value* truediv(Value* self, Value* other) {
    return mul(self, pow_val(other, -1.0));
}

static void topo_sort(Value* v, Value** sorted, size_t* count, Value** visited, size_t* n_visited) {
    for (size_t i = 0; i < *n_visited; i++)
        if (visited[i] == v) return;
    visited[(*n_visited)++] = v;

    for (size_t i = 0; i < v->n_children; i++)
        topo_sort(v->children[i], sorted, count, visited, n_visited);

    sorted[(*count)++] = v;
}

void backward(Value* self) {
    Value* sorted[256];
    Value* visited[256];
    size_t count = 0, n_visited = 0;

    topo_sort(self, sorted, &count, visited, &n_visited);

    self->grad = 1.0;
    for (size_t i = count; i-- > 0;)
        if (sorted[i]->_backward)
            sorted[i]->_backward(sorted[i]);
}

static const char* op_to_str(Op op) {
    switch (op) {
        case OP_ADD:  return "+";
        case OP_MUL:  return "*";
        case OP_POW:  return "**";
        case OP_RELU: return "ReLU";
        default:      return "";
    }
}

void print_value(const Value* self) {
    printf("Value(data=%.4f, grad=%.4f", self->data, self->grad);
    if (self->op != OP_NONE)
        printf(", op=%s", op_to_str(self->op));
    printf(")\n");
}

void free_value(Value* self) {
    if (!self) return;

    Value* sorted[256];
    Value* visited[256];
    size_t count = 0, n_visited = 0;

    topo_sort(self, sorted, &count, visited, &n_visited);

    for (size_t i = 0; i < count; i++) {
        free(sorted[i]->children);
        free(sorted[i]);
    }
}
