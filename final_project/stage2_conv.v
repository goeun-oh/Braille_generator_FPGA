`timescale 1ns / 1ps

// `define    CI           3  // Number of Channel Input 
// `define    CO           3 // Number of Channel Output
// `define    KX			5  // Number of Kernel X
// `define    KY			5  // Number of Kernel Y
// `define    W_BW         7  // BW of weight parameter
// `define    B_BW         6  // BW of bias parameter
// `define    M_BW         26 // I_F_BW * W_BW
// `define    AK_BW        31 // M_BW + log(KY*KX) Accum Kernel 
// `define    ACI_BW		33 // AK_BW + log (CI) Accum Channel Input
// `define    AB_BW        33 // ACI_BW + bias (#1). 
// `define    O_F_BW       33 // No Activation, So O_F_BW == AB_BW

// // `define    O_F_ACC_BW   27 // for demo, O_F_BW + log (CO)

// //pooling interface
// `define    ST2_Pool_IBW    19 // Stage2 Pooling input bitwidth
// `define    ST2_Pool_CI     3  // Number of Stage2 Pooling Channel Input
// `define    ST2_Pool_CO     3  // Number of Stage2 Pooling Channel Output
// `define    ST2_Pool_X      24 // Number of X (Input Channel)
// `define    ST2_Pool_Y      24 // Number of y (Input Channel)

// //convolution interface
// `define    ST2_Conv_IBW    19 // Conv Input Bitwidth
// `define    ST2_Conv_CI     3  // Number of Stage2 Conv Channel Input
// `define    ST2_Conv_CO     3  // Number of Stage2 Conv Channel Output
// `define    ST2_Conv_X      12 // Number of X (Conv Input Channel)
// `define    ST2_Conv_Y      12 // Number of y (Con Input Channel)
`include "stage2_defines_cnn_core.v"
module stage2_conv(
    // Clock & Reset
    clk             ,
    reset_n         ,
    i_in_valid      ,
    i_in_fmap       ,
    o_ot_valid      ,
    o_ot_fmap       
    );


//==============================================================================
// Input/Output declaration
//==============================================================================
input                                                                 clk         	;
input                                                                 reset_n     	;
input                                                                 i_in_valid  	;
input     signed [`ST2_Conv_CI * `ST2_Conv_IBW-1 : 0]  	              i_in_fmap    	;//3*(n bit) , 3ch에 대한 1point input
output                                                                o_ot_valid  	;
output    signed [`ST2_Conv_CO * (`O_F_BW-1)-1 : 0]  		          o_ot_fmap     ;//3*(n bit) , 3ch에 대한 1point output


    // 3 * (3 * 5 * 5) * (8bit)
    wire signed  [`ST2_Conv_CI*`ST2_Conv_CO*  `KX*`KY  *`W_BW -1 : 0] w_cnn_weight;
    wire signed  [`ST2_Conv_CI*`B_BW - 1  : 0]   w_cnn_bias;

    conv2_weight_rom u_weight_rom (
        .weight(w_cnn_weight) // 3x(3x5x5)
    );

    conv2_bias_rom u_bias_rom (
        .bias(w_cnn_bias) // 3x(3x5x5)
    );

    stage2_cnn_core u_stage2_cnn_core(
        .clk(clk)                     ,
        .reset_n(reset_n)             ,
        .i_cnn_weight(w_cnn_weight)   ,
        .i_cnn_bias(w_cnn_bias)                 ,
        .i_in_valid(i_in_valid)       ,
        .i_in_fmap(i_in_fmap)         ,
        .o_ot_valid(o_ot_valid)       ,
        .o_ot_fmap(o_ot_fmap)      
    );


endmodule
