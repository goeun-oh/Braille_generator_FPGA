`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/20 14:33:53
// Design Name: 
// Module Name: stage2_conv
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

`include "defines_cnn_core.v"

module stage2_conv(
    input                                                                                               clk     ,
    input                                                                                               reset_n ,

    input                                                                                               i_in_valid,

    //               3      *       12   *    12     *     (19bit)
    input     [`ST2_Conv_CI * `ST2_Conv_X*`ST2_Conv_Y*`ST2_Conv_IBW-1 : 0]  	                        i_in_fmap,
    output                                                                                              o_ot_valid,

    //               3      *       (12-5+1)=8        *       (12-5+1)=8       *  (33-1bit)    
    output    [`ST2_Conv_CO * (`ST2_Conv_X - `KX + 1) * (`ST2_Conv_Y - `KY + 1)*(`O_F_BW - 1) - 1 : 0]  o_ot_fmap    	
    );



    localparam X_MAX = `ST2_Conv_X - `KX + 1; // 8
    localparam Y_MAX = `ST2_Conv_Y - `KY + 1; // 8
    localparam TOTAL_POINTS = X_MAX * Y_MAX;

    reg [5:0] x_idx, y_idx;

    //    5  * 5   *      19bit
    reg [`KX * `KY * `ST2_Conv_IBW -1 :0] line_buffer_ch0;
    reg [`KX * `KY * `ST2_Conv_IBW -1 :0] line_buffer_ch1;
    reg [`KX * `KY * `ST2_Conv_IBW -1 :0] line_buffer_ch2;


    //    3 *  5*  5*     19bit
    reg [`CI*`KX*`KY*`ST2_Conv_IBW-1 : 0] patch;

   

    integer i, j, k, c;

    // c:0~2, i:0~4, j:0~4
    //(c*144 + (i+x) + (j+y)12)*19bit +: 19bit
    // input featuremap Channel 3개에 대하여 5x5를 sampling해서 patch에 넣기
    // patch 생성
    always @(*) begin
        for (c = 0; c < `CI; c = c + 1)
            for (i = 0; i < `KY; i = i + 1)
                for (j = 0; j < `KX; j = j + 1)
                    patch[(c*`KX*`KY + i*`KX + j)*`ST2_Conv_IBW +: `ST2_Conv_IBW] =
                        i_in_fmap[(c*`ST2_Conv_X*`ST2_Conv_Y + (y_idx+i)*`ST2_Conv_X + (x_idx+j))*`ST2_Conv_IBW +: `ST2_Conv_IBW];
    end

    cnn_core u_cnn_core(
        .clk(clk)                 ,
        .reset_n(reset_n)         ,
        .i_in_valid()             ,
        .i_in_fmap(patch)         ,
        .o_ot_valid()             ,
        .o_ot_fmap()      
    );


endmodule
