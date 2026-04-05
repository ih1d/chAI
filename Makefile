CC=gcc
CFLAGS=-Wall -O2 -Iinclude

SRC=$(wildcard src/*.c)
OBJ=$(SRC:.c=.o)

libchai.a: $(OBJ)
	ar rcs $@ $^

clean:
	rm -rf src/*.o libchai.a 
