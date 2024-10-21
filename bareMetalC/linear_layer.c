#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/resnet8_params.h"
#include "include/gemmini.h"
#include "include/gemmini_nn.h"



const int IN_CHANNELS = 1; //get size of fourth dimension
const int NB_CLASSES = sizeof(linear_w[0]) / sizeof(linear_w[0][0]); //get size of second dimension
const int IN_FEATURES = sizeof(linear_in[0]) / sizeof(linear_in[0][0]); //get size of second dimension of the array

const int BATCH_SIZE = sizeof(linear_in) / sizeof(linear_in[0]); //get size of fist dimension of the array
const int PADDING = 1;
const int STRIDE = 1;



const int NO_BIAS = 1;

const size_t I = BATCH_SIZE;
const size_t J = NB_CLASSES;
const size_t K = sizeof(linear_w) / sizeof(linear_w[0]);

int main() {
    #ifndef BAREMETAL
        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
        }
    #endif

    printf("NB_CLASSES = %d\n", NB_CLASSES);
    printf("IN_FEATURES = %d\n", IN_FEATURES);

    printf("BATCH_SIZE = %d\n", BATCH_SIZE);


    gemmini_flush(0);
    gemmini_config_multiplier(255, 16383);

    elem_t output_mat[BATCH_SIZE][NB_CLASSES];

    printf("Gemmini conv...\n");

    
    uint64_t start_gemmini = read_cycles();
    tiled_matmul_nn_auto(I, J, K,
        linear_in, linear_w, linear_b, output_mat,
        NO_ACTIVATION, 1.0 / 49,
        false,
        OS, false, "linear");


    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);

    printf("input:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int ich = 0; ich < IN_FEATURES; ich++) {
                printf("%d,", linear_in[batch][ich]);
            }
            printf("],");
        }
    printf("\n\n");

    printf("weights:\n");
        for (int batch = 0; batch < IN_FEATURES; batch++) {
            printf("[");
                for (int ich = 0; ich < NB_CLASSES; ich++) {
                    printf("%d,", linear_w[batch][ich]);
                }
            printf("],");
        }
    printf("\n\n");

    printf("bias:\n");
        for (int batch = 0; batch < NB_CLASSES; batch++) {
            printf("%d,", linear_b[batch]);
        }
    printf("\n\n");



    printf("output_mat:\n");
    for (int orow = 0; orow < BATCH_SIZE; orow++) {
        printf("[");
        for (int ocol = 0; ocol < NB_CLASSES; ocol++) {
            if(ocol == NB_CLASSES-1){
                printf("%d", output_mat[orow][ocol]);
            } else{
                printf("%d,", output_mat[orow][ocol]);
            }
        }
        printf("]\n");
    }
    printf("\n");


    return 0;
}
