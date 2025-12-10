#include <stdio.h>
#include <math.h>
#include <string.h>

#define LENGTH 3

void cambia_intero(int x){
    x = x+1;
}

int main(int argc, char** argv){
    int a = 100;
    int *pa = &a;
    int b = *pa;
    int *pb = pa + 1;

    return 0;
}