#include "../include/value.h"
#include <stdio.h>

int main(void) {
    Value* a = create_value(-4.0);
    Value* b = create_value(2.0);
    Value* c = add(a, b);
    Value* d = add(mul(a, b), pow_val(b, 3));
    c = add(c, add(c, create_value(1)));
    c = add(c, add(add(create_value(1), c), neg(a)));
    d = add(d, add(mul(d, create_value(2)), relu(add(b, a))));
    d = add(d, add(mul(create_value(3), d), relu(sub(b, a))));
    Value* e = sub(c, d);
    Value* f = pow_val(e, 2);
    Value* g = truediv(f, create_value(2));
    g = add(g, truediv(create_value(10), f));

    printf("Forward pass:\n");
    printf("g: "); print_value(g);

    backward(g);

    printf("\nAfter backward:\n");
    printf("a: "); print_value(a);
    printf("b: "); print_value(b);
    printf("c: "); print_value(c);
    printf("d: "); print_value(d);
    printf("e: "); print_value(e);
    printf("f: "); print_value(f);
    printf("g: "); print_value(g);

    free_value(g);
    return 0;
}
