/*******************************************************************************
Author: joohan.kim (https://blog.naver.com/chacagea)
Associated Filename: defines_cnn_core.vh
Purpose: c code to understand the CNN operation
License : https://github.com/matbi86/matbi_fpga_season_1/blob/main/LICENSE
Revision History: February 13, 2020 - initial release
*******************************************************************************/

`define    CI           3  // Number of Channel Input 
`define    CO           3 // Number of Channel Output
`define    I_F_BW       19  // Bit Width of Input Feature
`define    KX			5  // Number of Kernel X
`define    KY			5  // Number of Kernel Y
`define    W_BW         7  // BW of weight parameter
`define    B_BW         6  // BW of bias parameter
`define    M_BW         26 // I_F_BW * W_BW
`define    AK_BW        31 // M_BW + log(KY*KX) Accum Kernel 
`define    ACI_BW		33 // AK_BW + log (CI) Accum Channel Input
`define    AB_BW        33 // ACI_BW + bias (#1). 
`define    O_F_BW       33 // No Activation, So O_F_BW == AB_BW

// `define    O_F_ACC_BW   27 // for demo, O_F_BW + log (CO)

//pooling interface
`define    ST2_Pool_IBW    20 // Stage2 Pooling input bitwidth
`define    ST2_Pool_CI     3  // Number of Stage2 Pooling Channel Input
`define    ST2_Pool_CO     3  // Number of Stage2 Pooling Channel Output
`define    ST2_Pool_X      24 // Number of X (Input Channel)
`define    ST2_Pool_Y      24 // Number of y (Input Channel)

//convolution interface
`define    ST2_Conv_IBW    20 // Conv Input Bitwidth
`define    ST2_Conv_CI     3  // Number of Stage2 Conv Channel Input
`define    ST2_Conv_CO     3  // Number of Stage2 Conv Channel Output
`define    ST2_Conv_X      12 // Number of X (Conv Input Channel)
`define    ST2_Conv_Y      12 // Number of y (Con Input Channel)
