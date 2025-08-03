`timescale 1ns / 1ps
`include "defines_cnn_core.v"

`define IMG_SIZE                    28*28

//input interface still not defined clearly


module gray_filter(
        input   clk,
        input   reset_n,
        input   [`ISP_BW - 1 : 0]  one_px,              // 32 bit
        input   i_in_valid,                            
        output  [`ST1_I_F_BW - 1 : 0] grayed_one_px,    // 8 bit
        output  o_valid
    );

    localparam LATENCY = 1;

    wire [7:0] r_data;
    wire [7:0] g_data;
    wire [7:0] b_data;
    assign r_data = one_px[23:16];
    assign g_data = one_px[15: 8];
    assign b_data = one_px[ 7: 0];

    //16bit temp data
    reg  [15 : 0]  grayed_temp;
    
    always @(*) begin
        grayed_temp = r_data * 77 + g_data * 150 + b_data * 29;
    end

//==============================================================================
// Data Enable Signals 
//==============================================================================
    reg     [LATENCY-1 : 0] 	 r_valid;
    reg     [`ST1_I_F_BW - 1 : 0] r_grayed_one_px;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid   <= 0;
        end else begin
            r_valid[LATENCY-1]  <= i_in_valid;
        end
    end    

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_grayed_one_px <= 0;
        end else if(i_in_valid) begin
            r_grayed_one_px <= grayed_temp[15:8];
        end
    end

    assign grayed_one_px  = r_grayed_one_px;
    assign o_valid        = r_valid[LATENCY-1];

endmodule
