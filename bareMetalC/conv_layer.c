#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h> 
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/resnet8_params.h"



const int IN_ROW_DIM = sizeof(layer3_0_conv2_in[0]) / sizeof(layer3_0_conv2_in[0][0]); //get size of second dimension
const int IN_COL_DIM = sizeof(layer3_0_conv2_in[0][0]) / sizeof(layer3_0_conv2_in[0][0][0]); //get size of third dimension
const int IN_CHANNELS = sizeof(layer3_0_conv2_in[0][0][0]) / sizeof(elem_t);//get size of fourth dimension
const int OUT_CHANNELS = sizeof(layer3_0_conv2_w) / sizeof(layer3_0_conv2_w[0]); //get size of first dimension

const int BATCH_SIZE = sizeof(layer3_0_conv2_in) / sizeof(layer3_0_conv2_in[0]); //get size of fist dimension of the array
const int KERNEL_DIM = sizeof(layer3_0_conv2_w[0]) / sizeof(layer3_0_conv2_w[0][0]);
const int PADDING = 1;
const int STRIDE = 1;

const int NO_BIAS = 1;

const int OUT_ROW_DIM = ((IN_ROW_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1);
const int OUT_COL_DIM = ((IN_COL_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1);
const int PATCH_SIZE = (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS);
const int N_PATCHES = (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM);

const bool FAST = 0;



bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            return false;
    return true;
}

void compute_errors(double *mae, double *mse, double *max_ae, elem_t *array1, elem_t *array2, size_t length) {
    double sum_absolute_errors_mae = 0.0;
    double sum_squared_errors_mse = 0.0;
    double max_error = 0;

    for (size_t i = 0; i < length; i++) {
        double diff = fabs((double)array1[i] - (double)array2[i]); //diff is used to compute all 3 errors

        sum_absolute_errors_mae += diff;

        sum_squared_errors_mse += diff * diff;

        if (diff > max_error){
            max_error = diff;
        }
    }

    *mae = sum_absolute_errors_mae / length;
    *mse = sum_squared_errors_mse / length;
    *max_ae = max_error;
}

int compute_leading_zeros(double fractional, int decimal_places){
    int nb_zeros = 0;
    while(nb_zeros < decimal_places){
        fractional = fractional * 10.0;
        // int whole_part = (int)fractional;
        // double fract = fabs(fractional - whole_part);
        // int fraction_part = (int)(fract * 100);
        //printf("fractional = %d\n", (int)fractional);
        if((int)fractional > 0){
            return nb_zeros;
        }
        nb_zeros++;
    }
    return decimal_places;
}

int my_pow(int base, int exponent) {
    int result = 1;
    for (int i = 0; i < exponent; i++) {
        result *= base;  // Multiply the base by itself
    }

    return result;
}

void split_double(double num, int *whole_part, int *fraction_part, int *leading_zeros, int decimal_places) {
    *whole_part = (int)num;  // Get the integer part (left of the decimal)
    
    double fractional = fabs(num - *whole_part);  // Get the fractional part (right of the decimal)

    *leading_zeros = compute_leading_zeros(fractional, decimal_places);

    *fraction_part = (int)(fractional * my_pow(10, decimal_places));  // Convert the fractional part to an integer
}

void print_double(int whole, int fractional, int nb_of_leading_zeros) {
    // Print the whole part
    printf("%d.", whole);
    
    // Print leading zeros for the fractional part
    for (int i = 0; i < nb_of_leading_zeros; i++) {
        putchar('0');
    }
    
    // Print the fractional part
    printf("%d\n", fractional);
}

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

    printf("IN_ROW_DIM = %d\n", IN_ROW_DIM);
    printf("IN_COL_DIM = %d\n", IN_COL_DIM);
    printf("IN_CHANNELS = %d\n", IN_CHANNELS);
    printf("OUT_CHANNELS = %d\n", OUT_CHANNELS);

    printf("BATCH_SIZE = %d\n", BATCH_SIZE);
    printf("KERNEL_DIM = %d\n", KERNEL_DIM);
    
    printf("OUT_ROW_DIM = %d\n", OUT_ROW_DIM);
    printf("OUT_COL_DIM = %d\n", OUT_COL_DIM);
    printf("PATCH_SIZE = %d\n", PATCH_SIZE);
    printf("N_PATCHES = %d\n", N_PATCHES);

    gemmini_flush(0);
    gemmini_config_multiplier(255, 16383);
    

    printf("Input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM, IN_COL_DIM);
    printf("Output dimensions (rows by columns): %u by %u\n\n", OUT_ROW_DIM, OUT_COL_DIM);
    
    //static elem_t input[BATCH_SIZE][IN_ROW_DIM][IN_COL_DIM][IN_CHANNELS];
    //static elem_t weights[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS];
    acc_t bias[OUT_CHANNELS];
    //static elem_t output[BATCH_SIZE][OUT_ROW_DIM][OUT_COL_DIM][OUT_CHANNELS];

    
    // printf("Randomize inputs...\n");
    // init_pseudo_random(&input[0][0][0][0], sizeof(input) / sizeof(elem_t));

    // printf("Randomize weights...\n");
    // init_pseudo_random(&weights[0][0][0][0], sizeof(weights) / sizeof(elem_t));

    if (NO_BIAS)
        init_zeros_acc(&bias[0], sizeof(bias) / sizeof(acc_t));


    elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS];
    elem_t output_mat[N_PATCHES][OUT_CHANNELS];

    printf("Flatten weights...\n");
    flatten_weights(OUT_CHANNELS, KERNEL_DIM, IN_CHANNELS,
            PATCH_SIZE,
            layer3_0_conv2_w,
            weights_mat);

    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();
    tiled_conv_auto(
        BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_ROW_DIM, OUT_COL_DIM,
        STRIDE, 1, 1, PADDING, KERNEL_DIM,
        false, false, false, false, false,

        (elem_t*)layer3_0_conv2_in,
        (elem_t*)weights_mat,
        NO_BIAS ? NULL : (acc_t*)layer3_0_conv2_b,
        (elem_t*)output_mat,

        NO_ACTIVATION, 1.0 / 452,
        0, 0, 0,

        WS);
    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);
    

    double mae, mse, max_ae;
    int whole_mae, fraction_mae, whole_mse, fraction_mse, whole_max_ae, fraction_max_ae;
    int leading_zeros_mae, leading_zeros_mse, leading_zeros_max_ae;

    
    compute_errors(&mae, &mse, &max_ae, &layer3_0_conv2_out[0][0][0][0], &output_mat[0][0], sizeof(layer3_0_conv2_out) / sizeof(elem_t));

    split_double(mae, &whole_mae, &fraction_mae, &leading_zeros_mae, 5);
    split_double(mse, &whole_mse, &fraction_mse, &leading_zeros_mse, 5);
    split_double(max_ae, &whole_max_ae, &fraction_max_ae, &leading_zeros_max_ae, 0);

    printf("Mean Absolute Error (MAE) = ");
    print_double(whole_mae, fraction_mae, leading_zeros_mae);

    printf("Mean Squared Error (MSE) = ");
    print_double(whole_mse, fraction_mse, leading_zeros_mse);

    printf("Maximum Absolute Error (Max AE) = ");
    print_double(whole_max_ae, fraction_max_ae, leading_zeros_max_ae);

    // printf("input:\n");
    //     for (int batch = 0; batch < BATCH_SIZE; batch++) {
    //         printf("[");
    //         for (int irow = 0; irow < IN_ROW_DIM; irow++) {
    //             printf("[");
    //             for (int icol = 0; icol < IN_COL_DIM; icol++) {
    //                 printf("[");
    //                 for (int ich = 0; ich < IN_CHANNELS; ich++) {
    //                     printf("%d,", layer3_0_conv2_in[batch][irow][icol][ich]);
    //                 }
    //                 printf("],");
    //             }
    //             printf("],\n");
    //         }
    //         printf("],");
    //     }
    // printf("\n\n");

    // printf("weights:\n");
    //     for (int och = 0; och < OUT_CHANNELS; och++) {
    //         printf("[");
    //         for (int wker = 0; wker < KERNEL_DIM; wker++) {
    //             printf("[");
    //             for (int hker = 0; hker < KERNEL_DIM; hker++) {
    //                 printf("[");
    //                 for (int ich = 0; ich < IN_CHANNELS; ich++) {
    //                     printf("%d,", layer3_0_conv2_w[och][wker][hker][ich]);
    //                 }
    //                 printf("],");
    //             }
    //             printf("],\n");
    //         }
    //         printf("],");
    //     }
    // printf("\n\n");

    // printf("bias:\n");
    //     for (int och = 0; och < OUT_CHANNELS; och++) {
    //         printf("%d,", bias[och]);
    //     }
    // printf("\n\n");


    if(!FAST){
        printf("output_mat:\n");
        for (int orow = 0; orow < BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM; orow++) {
            printf("[");
            for (int och = 0; och < OUT_CHANNELS; och++) {
                if(och == OUT_CHANNELS-1){
                    printf("%d", output_mat[orow][och]);
                } else{
                    printf("%d,", output_mat[orow][och]);
                }
            }
            printf("]\n");
        }
        printf("\n");
    }
    
    return 0;
}
