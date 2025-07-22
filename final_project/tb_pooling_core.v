`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/20 19:21:31
// Design Name: 
// Module Name: tb_pooling_core
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
module tb_pooling_core();


    reg                                                                                   clk         	;
    reg                                                                                   reset_n     	;
    reg                                                                                   i_in_valid  	;
    reg      [`ST2_Pool_CI * `ST2_Pool_IBW - 1 : 0]                                       i_in_fmap    	;//3*(19bit) , 3ch에 대한 1point input
    wire                                                                                  o_ot_valid  	;
    wire     [`ST2_Pool_CI * `ST2_Pool_IBW - 1 : 0]                                       o_ot_fmap     ;


    stage2_pooling_core dut(
        .clk        (clk),  
        .reset_n    (reset_n),  
        .i_in_valid (i_in_valid),  
        .i_in_fmap  (i_in_fmap),   
        .o_ot_valid (o_ot_valid), 	    
        .o_ot_fmap  (o_ot_fmap)        
    );

     always #5 clk = ~clk;
    integer i;
    integer j;
    integer idx;
    integer r, c;
    initial begin
        clk = 0;
        reset_n = 1;
        i_in_valid = 0;
        #10;
        reset_n = 0;
        #10;
        reset_n = 1;
        #25;
        #1;
        i_in_valid = 1;
        for (i = 0; i < `ST2_Pool_X * `ST2_Pool_Y; i = i + 1) begin
            i_in_fmap[0               +: `ST2_Pool_IBW] = i[(`ST2_Pool_IBW) - 1 : 0];  // 입력값 변화
            i_in_fmap[1*`ST2_Pool_IBW +: `ST2_Pool_IBW] = i[(`ST2_Pool_IBW) - 1 : 0];  // 입력값 변화
            i_in_fmap[2*`ST2_Pool_IBW +: `ST2_Pool_IBW] = i[(`ST2_Pool_IBW) - 1 : 0];  // 입력값 변화
            #10; // 클럭 주기만큼 대기
        end   
        #10;

        // #100;
        // $display("==== i_in_fmap 24x24 ====");
        // for (r = 0; r < 24; r = r + 1) begin
        //     for (c = 0; c < 24; c = c + 1) begin
        //         $write("(%6d)", i_in_fmap[(r*24 + c)*19 +: 19]);
        //         if ((c % 2) == 1) $write(" | ");  // 세로 블록 경계
        //     end
        //     $write("\n");
        //     if ((r % 2) == 1) begin
        //         for (c = 0; c < 24; c = c + 1) begin
        //             $write("---------");
        //             if ((c % 2) == 1) $write("+");
        //         end
        //         $write("\n");
        //     end
        // end

        // $display("==== o_ot_fmap 12x12 ====");
        // for (r = 0; r < 12; r = r + 1) begin
        //     for (c = 0; c < 12; c = c + 1) begin
        //         idx = r * 12 + c;
        //         $write("(%6d) ", o_ot_fmap[idx*19 +: 19]);
        //     end
        //     $write("\n");
        // end
        $finish;
    end
    // always @(posedge clk) begin
    //     $display("time %t: , row=%d, col=%d", $time, dut.row, dut.col);
    //     $display("max : %d", dut.r_o_pooling);
    // end    
endmodule
