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

const int NO_BIAS = 0;

const size_t I = BATCH_SIZE;
const size_t J = NB_CLASSES;
const size_t K = sizeof(linear_w) / sizeof(linear_w[0]);

const bool FAST = 1;

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
    gemmini_config_multiplier(255 - appr_level[7], 16383);

    elem_t output_mat[BATCH_SIZE][NB_CLASSES];

    printf("Gemmini matmul...\n");
    uint64_t start_gemmini = read_cycles();
    tiled_matmul_nn_auto(I, J, K,
        linear_in, linear_w, linear_b, output_mat,
        NO_ACTIVATION, 1.0 / 150,
        false,
        OS, false, "linear");


    uint64_t end_gemmini = read_cycles();
    printf("Gemmini matmul took %llu cycles\n\n", end_gemmini - start_gemmini);

    int max_index[BATCH_SIZE];
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int i = 1; i < NB_CLASSES; i++) {
            if (output_mat[batch][i] > output_mat[batch][max_index[batch]]) {
                max_index[batch] = i;
            }
        }
    }

    if(!FAST){
        printf("output_mat:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int class = 0; class < NB_CLASSES; class++) {
                if(class == NB_CLASSES-1){
                    printf("%d", output_mat[batch][class]);
                } else{
                    printf("%d,", output_mat[batch][class]);
                }
            }
            printf("]\n");
        }
        printf("\n");
    }
    
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        printf("max_index[%d] = %d\n", batch, max_index[batch]);
        if(max_index[batch] != labels[batch]) {
            exit(1);
        }
    }

    double mae, mse, max_ae;
    int whole_mae, fraction_mae, whole_mse, fraction_mse, whole_max_ae, fraction_max_ae;
    int leading_zeros_mae, leading_zeros_mse, leading_zeros_max_ae;

    uint64_t start_error_computation = read_cycles();
    compute_errors(&mae, &mse, &max_ae, &linear_out[0][0], &output_mat[0][0], sizeof(linear_out) / sizeof(elem_t));
    uint64_t end_error_computation = read_cycles();
    printf("Error compuation took %llu cycles\n", end_gemmini - start_gemmini);
    
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
    //         for (int in_ft = 0; in_ft < IN_FEATURES; in_ft++) {
    //             printf("%d,", linear_in[batch][in_ft]);
    //         }
    //         printf("],");
    //     }
    // printf("\n\n");


    // printf("weights:\n");
    //     for (int in_ft = 0; in_ft < IN_FEATURES; in_ft++) {
    //         printf("[");
    //             for (int class = 0; class < NB_CLASSES; class++) {
    //                 printf("%d,", linear_w[in_ft][class]);
    //             }
    //         printf("],");
    //     }
    // printf("\n\n");


    // printf("bias:\n");
    //     for (int class = 0; class < NB_CLASSES; class++) {
    //         printf("%d,", linear_b[class]);
    //     }
    // printf("\n\n");

    return 0;
}
