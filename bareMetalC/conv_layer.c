#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "conv_layer_params.h"



#define IN_ROW_DIM 32
#define IN_COL_DIM 32
#define IN_CHANNELS 3
#define OUT_CHANNELS 16


#define BATCH_SIZE 1
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 1



#define NO_BIAS true

#define OUT_ROW_DIM ((IN_ROW_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define OUT_COL_DIM ((IN_COL_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM)



void flatten_weights(int out_channels, int kernel_dim, int in_channels,
        int patch_size,
        elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
        elem_t weights_mat[patch_size][out_channels]) {

    assert(patch_size == kernel_dim * kernel_dim * in_channels);

    for (int outc = 0; outc < out_channels; outc++) {
        for (int krow = 0; krow < kernel_dim; krow++) {
            for (int kcol = 0; kcol < kernel_dim; kcol++) {
                for (int inc = 0; inc < in_channels; inc++) {
                    int wmatrow = krow * kernel_dim * in_channels +
                        kcol * in_channels +
                        inc;

                    weights_mat[wmatrow][outc] =
                        weights[outc][krow][kcol][inc];
                }
            }
        }
    }
}



void init_zeros_acc(acc_t * buf, int len) {
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        *ptr = 0;
    }
}



int main() {
    #ifndef BAREMETAL
        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
        }
    #endif

    gemmini_flush(0);
    gemmini_config_multiplier(255, 16383);
    

    printf("Input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM, IN_COL_DIM);
    printf("Output dimensions (rows by columns): %u by %u\n\n", OUT_ROW_DIM, OUT_COL_DIM);

    //static elem_t input[BATCH_SIZE][IN_ROW_DIM][IN_COL_DIM][IN_CHANNELS];
    //static elem_t weights[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS];
    static acc_t bias[OUT_CHANNELS];
    //static elem_t output[BATCH_SIZE][OUT_ROW_DIM][OUT_COL_DIM][OUT_CHANNELS];

    
    // printf("Randomize inputs...\n");
    // init_pseudo_random(&input[0][0][0][0], sizeof(input) / sizeof(elem_t));

    // printf("Randomize weights...\n");
    // init_pseudo_random(&weights[0][0][0][0], sizeof(weights) / sizeof(elem_t));

    if (NO_BIAS)
        init_zeros_acc(&bias[0], sizeof(bias) / sizeof(acc_t));


    static elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS];
    static elem_t output_mat[N_PATCHES][OUT_CHANNELS];

    printf("Flatten weights...\n");
    flatten_weights(OUT_CHANNELS, KERNEL_DIM, IN_CHANNELS,
            PATCH_SIZE,
            conv_1_w,
            weights_mat);

    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();
    tiled_conv_auto(
        BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_ROW_DIM, OUT_COL_DIM,
        STRIDE, 1, 1, PADDING, KERNEL_DIM,
        false, false, false, false, false,

        (elem_t*)conv_1_in,
        (elem_t*)weights_mat,
        NO_BIAS ? NULL : (acc_t*)bias,
        (elem_t*)output_mat,

        NO_ACTIVATION, 1.0 / 162, 0, 0, 0,

        WS);
    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);

    printf("input:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int irow = 0; irow < IN_ROW_DIM; irow++) {
                printf("[");
                for (int icol = 0; icol < IN_COL_DIM; icol++) {
                    printf("[");
                    for (int ich = 0; ich < IN_CHANNELS; ich++) {
                        printf("%d,", conv_1_in[batch][irow][icol][ich]);
                    }
                    printf("],");
                }
                printf("],\n");
            }
            printf("],");
        }
    printf("\n\n");

    printf("weights:\n");
        for (int batch = 0; batch < OUT_CHANNELS; batch++) {
            printf("[");
            for (int irow = 0; irow < KERNEL_DIM; irow++) {
                printf("[");
                for (int icol = 0; icol < KERNEL_DIM; icol++) {
                    printf("[");
                    for (int ich = 0; ich < IN_CHANNELS; ich++) {
                        printf("%d,", conv_1_w[batch][irow][icol][ich]);
                    }
                    printf("],");
                }
                printf("],\n");
            }
            printf("],");
        }
    printf("\n\n");

    printf("bias:\n");
        for (int batch = 0; batch < OUT_CHANNELS; batch++) {
            printf("%d,", bias[batch]);
        }
    printf("\n\n");



    printf("output_mat:\n");
    for (int orow = 0; orow < BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM; orow++) {
        printf("[");
        for (int ocol = 0; ocol < OUT_CHANNELS; ocol++) {
            if(ocol == OUT_CHANNELS-1){
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
