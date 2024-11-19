#include <include/gemmini_params.h>
#include <stdbool.h>

// conv_1_w[in_channel][kernel_dim][kernel_dim][out_channel] = [in_channel*kernel_dim*kernel_dim][out_channel]
// conv_1_in[in_row_dim][in_col_dim][in_channels]
// conv_1_out_pooled[out_dim_pooled][out_dim_pooled][out_channels] 

// N_PATCHES = (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM);
// PATCH_SIZE = (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS);
// Output Dimension = (in_dim - kernel_size + 2*Padding)/stride + 1
static const elem_t conv_1_w[363][96] row_align(1);
static const acc_t conv_1_b[96] row_align_acc(1);
static elem_t conv_1_in row_align(1); //Not used, the first input is the image
static elem_t conv_1_out_pooled[4][27][27][96] row_align(1);
static const struct ConvParams conv_1_params = {.batch_size=4, .in_row_dim=227, .in_col_dim=227, .kernel_size=11, .in_channels=3,
 .out_channels=96, .stride=4, .padding=0, .bias=1, .depthwise=0, .out_row_dim=55, .out_col_dim=55, .n_patches=51529, .patch_size=363,
 .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=27, .output_scale=(1.0 / (1 << 8))};


static const elem_t conv_2_w[2400][256] row_align(1);
static const acc_t conv_2_b[256] row_align_acc(1);
static elem_t conv_2_in row_align(1); 
static elem_t conv_2_out_pooled[4][13][13][256] row_align(1);
static const struct ConvParams conv_2_params = {.batch_size=4, .in_row_dim=27, .in_col_dim=27, .kernel_size=5, .in_channels=96,
 .out_channels=256, .stride=1, .padding=2, .bias=1, .depthwise=0, .out_row_dim=27, .out_col_dim=27, .n_patches=2916, .patch_size=2400,
 .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=13, .output_scale=(1.0 / (1 << 8))};


static const elem_t conv_3_w[2304][384] row_align(1);
static const acc_t conv_3_b[384] row_align_acc(1);
static elem_t conv_3_in row_align(1); 
static elem_t conv_3_out[4][13][13][384] row_align(1);
static const struct ConvParams conv_3_params = {.batch_size=4, .in_row_dim=13, .in_col_dim=13, .kernel_size=3, .in_channels=256,
 .out_channels=384, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_row_dim=13, .out_col_dim=13, .n_patches=676, .patch_size=2304,
 .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=(1.0 / (1 << 8))};

// N_PATCHES = (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM);
// PATCH_SIZE = (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS);

static const elem_t conv_4_w[3456][384] row_align(1);
static const acc_t conv_4_b[384] row_align_acc(1);
static elem_t conv_4_in row_align(1); 
static elem_t conv_4_out[4][13][13][384] row_align(1);
static const struct ConvParams conv_4_params = {.batch_size=4, .in_row_dim=13, .in_col_dim=13, .kernel_size=3, .in_channels=384,
 .out_channels=384, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_row_dim=13, .out_col_dim=13, .n_patches=676, .patch_size=3456,
 .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=(1.0 / (1 << 8))};

static const elem_t conv_5_w[3456][256] row_align(1);
static const acc_t conv_5_b[256] row_align_acc(1);
static elem_t conv_5_in row_align(1); 
static elem_t conv_5_out_pooled[4][6][6][256] row_align(1);
static const struct ConvParams conv_5_params = {.batch_size=4, .in_row_dim=13, .in_col_dim=13, .kernel_size=3, .in_channels=384,
 .out_channels=256, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_row_dim=13, .out_col_dim=13, .n_patches=144, .patch_size=3456,
 .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=6, .output_scale=(1.0 / (1 << 8))};

static const elem_t fc_6_w[9216][4096] row_align(1)
static const acc_t fc_6_b[4096] row_align_acc(1)
static elem_t fc_6_out[4][4096] row_align(1);
static const struct FcParams fc_6_params = {.batch_size=4, .in_features=9216, .out_features=4096, .bias=1, .output_scale=(1.0 / (1 << 9)), .I=4, .J=4096, .K=9216};

static const elem_t fc_7_w[4096][4096] row_align(1)
static const acc_t fc_7_b[4096] row_align_acc(1)
static elem_t fc_7_out[4][4096] row_align(1);
static const struct FcParams fc_7_params = {.batch_size=4, .in_features=4096, .out_features=4096, .bias=1, .output_scale=(1.0 / (1 << 9)), .I=4, .J=4096, .K=4096};

static const elem_t fc_9_w[4096][1000] row_align(1)
static const acc_t fc_9_b[1000] row_align_acc(1)
static elem_t fc_9_out[4][1000] row_align(1);
static const struct FcParams fc_9_params = {.batch_size=4, .in_features=4096, .out_features=1000, .bias=1, .output_scale=(1.0 / (1 << 9)), .I=4, .J=1000, .K=9216};