
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/21 16:09:05
// Design Name: 
// Module Name: max_pooling
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
`include "defines_cnn_core.v"

module stage3_max_pooling(
    input wire clk,
    input wire reset_n,

    input wire i_Relu_valid,
    input wire [`stage3_CI * `ST3_IF_BW - 1: 0] i_in_Relu,

    output wire o_ot_valid,
    output wire [`pool_CO * `ST3_OF_BW-1:0] o_ot_pool
    );
    localparam LATENCY = 2;

    wire [`linebuf_CO-1 : 0] w_ot_valid;
    // 3 * 2 * 2 * 32
    wire [`linebuf_CO * `POOL_K*`POOL_K*`ST3_IF_BW-1:0] w_ot_window;

    // //디버깅 용
    // (* mark_debug = "true" *) reg [`ST3_IF_BW-1:0] w_ot_window00;
    // (* mark_debug = "true" *) reg [`ST3_IF_BW-1:0] w_ot_window01;
    // (* mark_debug = "true" *) reg [`ST3_IF_BW-1:0] w_ot_window02;
    // (* mark_debug = "true" *) reg [`ST3_IF_BW-1:0] w_ot_window03;
    // reg [`ST3_IF_BW-1:0] w_ot_window10;
    // reg [`ST3_IF_BW-1:0] w_ot_window11;
    // reg [`ST3_IF_BW-1:0] w_ot_window12;
    // reg [`ST3_IF_BW-1:0] w_ot_window13;
    // reg [`ST3_IF_BW-1:0] w_ot_window20;
    // reg [`ST3_IF_BW-1:0] w_ot_window21;
    // reg [`ST3_IF_BW-1:0] w_ot_window22;
    // reg [`ST3_IF_BW-1:0] w_ot_window23;

    // always @(posedge clk, negedge reset_n) begin
    //     if (!reset_n) begin
    //         w_ot_window00 <= 0;
    //         w_ot_window01 <= 0;
    //         w_ot_window02 <= 0;
    //         w_ot_window03 <= 0;
    //         w_ot_window10 <= 0;
    //         w_ot_window11 <= 0;
    //         w_ot_window12 <= 0;
    //         w_ot_window13 <= 0;
    //         w_ot_window20 <= 0;
    //         w_ot_window21 <= 0;
    //         w_ot_window22 <= 0;
    //         w_ot_window23 <= 0;
    //     end else begin
    //         w_ot_window00 <= w_ot_window[0*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window01 <= w_ot_window[1*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window02 <= w_ot_window[2*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window03 <= w_ot_window[3*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window10 <= w_ot_window[4*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window11 <= w_ot_window[5*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window12 <= w_ot_window[6*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window13 <= w_ot_window[7*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window20 <= w_ot_window[8*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window21 <= w_ot_window[9*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window22 <= w_ot_window[10*`ST3_IF_BW +: `ST3_IF_BW];
    //         w_ot_window23 <= w_ot_window[11*`ST3_IF_BW +: `ST3_IF_BW];
    //     end
    // end


    // 48 * 32
    wire [`pool_CO * `ST3_OF_BW-1:0] w_ot_pool;
    reg  [`pool_CO * `ST3_OF_BW-1:0] w_ot_flat;
    reg r_pooling_valid;


//==============================================================================
// Data Enable Signals 
//==============================================================================
wire    [LATENCY-1 : 0] 	ce;
reg     [LATENCY-1 : 0] 	r_valid;
always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_valid   <= 0;
    end else begin
        r_valid[LATENCY-2]  <= &w_ot_valid;
        r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
    end
end

assign	ce = r_valid;


    genvar line_inst;
    generate
        for (line_inst = 0; line_inst < `stage3_CI ; line_inst = line_inst + 1) begin
            wire [`ST3_IF_BW - 1: 0] w_in_pixel = i_in_Relu[line_inst * `ST3_IF_BW +: `ST3_IF_BW];
            stage3_line_buffer U_line_buffer(
                .clk(clk),
                .reset_n(reset_n),
                .i_in_valid(i_Relu_valid),
                .i_in_pixel(w_in_pixel),
                .o_window_valid(w_ot_valid[line_inst]),
                .o_window(w_ot_window[line_inst * `POOL_K * `POOL_K * `ST3_IF_BW +: `POOL_K * `POOL_K * `ST3_IF_BW])
            );
        end
    endgenerate

    genvar pool_inst;
    generate
        for (pool_inst = 0; pool_inst < `pool_CI ; pool_inst = pool_inst + 1) begin
            stage3_max_pool_2x2 U_max_pool (
                .i00(w_ot_window[pool_inst * `POOL_K * `POOL_K * `ST3_IF_BW +: `ST3_IF_BW]),
                .i01(w_ot_window[(pool_inst * `POOL_K * `POOL_K + 1) * `ST3_IF_BW +: `ST3_IF_BW]),
                .i10(w_ot_window[(pool_inst * `POOL_K * `POOL_K + 2) * `ST3_IF_BW +: `ST3_IF_BW]),
                .i11(w_ot_window[(pool_inst * `POOL_K * `POOL_K + 3) * `ST3_IF_BW +: `ST3_IF_BW]),
                .o_max(w_ot_pool[pool_inst * `ST3_OF_BW +: `ST3_OF_BW])
            );
        end
    endgenerate


    reg [`pool_CO * `ST3_OF_BW-1:0] r_pool_result;
    // 1클럭 pipelining
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_pool_result <= 0;
        end else if (&w_ot_valid) begin
            r_pool_result <= w_ot_pool;
        end 
    end



    // 디버깅용
    reg [`ST3_OF_BW -1 : 0]r_o_ot_flat [0:`pool_CO-1];
    integer i;
    always @(posedge clk, negedge reset_n) begin
        if (!reset_n) begin
            w_ot_flat <= 0;
            r_pooling_valid <= 0;
        end else if (r_valid[LATENCY-2]) begin
            w_ot_flat <= r_pool_result;
            // 디버깅용 시작
            for(i = 0; i< `pool_CO ; i = i + 1) begin
                r_o_ot_flat[i] <= r_pool_result[i * `ST3_OF_BW +: `ST3_OF_BW];    
            end
            // 디버깅용 끝
            r_pooling_valid <= 1;
        end 
        else begin
            r_pooling_valid <= 0;
        end
    end

    // (* mark_debug = "true" *) reg [`ST3_OF_BW -1 : 0] d_ot_flat0;
    // (* mark_debug = "true" *) reg [`ST3_OF_BW -1 : 0] d_ot_flat1;
    // (* mark_debug = "true" *) reg [`ST3_OF_BW -1 : 0] d_ot_flat2;
    // always @(posedge clk, negedge reset_n) begin
    //     if (!reset_n) begin
    //         d_ot_flat0 <= 0;
    //         d_ot_flat1 <= 0;
    //         d_ot_flat2 <= 0;
    //     end else begin
    //         d_ot_flat0 <= r_pool_result[0+:`ST3_OF_BW];
    //         d_ot_flat1 <= r_pool_result[`ST3_OF_BW+:`ST3_OF_BW];
    //         d_ot_flat2 <= r_pool_result[2*`ST3_OF_BW+:`ST3_OF_BW];
    //     end
    // end

    assign o_ot_pool = w_ot_flat;
    assign o_ot_valid = r_valid[LATENCY-1];

endmodule

module stage3_max_pool_2x2 (
    input  [`ST3_OF_BW-1:0] i00, i01, i10, i11,
    output [`ST3_OF_BW-1:0] o_max
);
    wire [`ST3_OF_BW-1:0] max0 = ($signed(i00) > $signed(i01)) ? i00 : i01;
    wire [`ST3_OF_BW-1:0] max1 = ($signed(i10) > $signed(i11)) ? i10 : i11;
    assign o_max = ($signed(max0) > $signed(max1)) ? max0 : max1;
endmodule