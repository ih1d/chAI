#ifndef VALUE_H
#define VALUE_H

#include <stdlib.h>

typedef enum {
    OP_NONE,
    OP_ADD,
    OP_MUL,
    OP_POW,
    OP_RELU,
} Op;

typedef struct Value {
    double data;
    double grad;
    struct Value** children;
    size_t n_children;
    Op op;
    void (*_backward)(struct Value* self);
} Value;

Value* create_value(double data);

Value* init_value(Value* self, double data, struct Value** children, size_t children_size, Op op);

/* Operations */
Value* add(Value* self, Value* other);
Value* mul(Value* self, Value* other);
Value* pow_val(Value* self, double exponent);
Value* relu(Value* self);
Value* neg(Value* self);
Value* sub(Value* self, Value* other);
Value* truediv(Value* self, Value* other);

void backward(Value* self);

void print_value(const Value* self);
void free_value(Value* self);

#endif
