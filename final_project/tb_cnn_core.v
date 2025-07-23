`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/22 14:49:35
// Design Name: 
// Module Name: tb_cnn_core
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

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
`define    ST2_Pool_IBW    19 // Stage2 Pooling input bitwidth
`define    ST2_Pool_CI     3  // Number of Stage2 Pooling Channel Input
`define    ST2_Pool_CO     3  // Number of Stage2 Pooling Channel Output
`define    ST2_Pool_X      24 // Number of X (Input Channel)
`define    ST2_Pool_Y      24 // Number of y (Input Channel)

//convolution interface
`define    ST2_Conv_IBW    19 // Conv Input Bitwidth
`define    ST2_Conv_CI     3  // Number of Stage2 Conv Channel Input
`define    ST2_Conv_CO     3  // Number of Stage2 Conv Channel Output
`define    ST2_Conv_X      12 // Number of X (Conv Input Channel)
`define    ST2_Conv_Y      12 // Number of y (Con Input Channel)


module tb_cnn_core();


    reg                                                   clk         	;
    reg                                                   reset_n     	;
    reg                                                   i_in_valid  	;
    reg     [`ST2_Conv_CI*`KX*`KY*`ST2_Conv_IBW-1 : 0]    i_in_fmap     ;
    wire                                                  o_ot_valid  	;
    wire    [`ST2_Conv_CO*(`O_F_BW-1)-1 : 0]  		      o_ot_fmap     ;

    cnn_core dut(
        .clk          (clk),   
        .reset_n      (reset_n),   
        .i_in_valid   (i_in_valid),   
        .i_in_fmap    (i_in_fmap),   
        .o_ot_valid   (o_ot_valid),   
        .o_ot_fmap    (o_ot_fmap)        
    );

    


endmodule
