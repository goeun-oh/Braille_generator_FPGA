`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/21 16:09:05
// Design Name: 
// Module Name: cnn_core
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
`include "stage3_defines_cnn_core.vh"

module stage3_cnn_core(
    input clk,
    input reset_n,

    // pooling valid
    input i_in_valid,
    // 48 개 중 0, 16, 32
    input [`acc_CO * `ACC_BW-1:0] o_ot_ci_acc,
    
    output o_ot_valid,
    output [`core_CO * `OUT_BW -1:0] o_ot_result
    );
    // bias
    localparam LATENCY = 1;
    reg signed [`BIAS_BW-1:0] bias_mem[0:2];
    

    //reg signed [`CO * `OUT_BW -1:0] w_ot_result;
    reg signed [`OUT_BW -1:0] w_ot_result[0:2];
    // reg signed [`OUT_BW -1:0] w_ot_result1;
    // reg signed [`OUT_BW -1:0] w_ot_result2;

    reg signed [`core_CO * `OUT_BW -1:0] r_ot_result;

    // (* mark_debug = "true" *) reg signed [`OUT_BW -1:0] d_ot_result0;
    // (* mark_debug = "true" *) reg signed [`OUT_BW -1:0] d_ot_result1;
    // (* mark_debug = "true" *) reg signed [`OUT_BW -1:0] d_ot_result2;
    // always @(posedge clk, negedge reset_n) begin
    //     if (!reset_n) begin
    //         d_ot_result0 <= 0;
    //         d_ot_result1 <= 0;
    //         d_ot_result2 <= 0;
    //     end else begin
    //         d_ot_result0 <= r_ot_result[0+:`OUT_BW];
    //         d_ot_result1 <= r_ot_result[`OUT_BW+:`OUT_BW];
    //         d_ot_result2 <= r_ot_result[2*`OUT_BW+:`OUT_BW];
    //     end
    // end


    reg  signed   [LATENCY - 1 : 0]         r_valid;

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid   <= 0;
        end else begin
            r_valid[LATENCY - 1]  <= i_in_valid;
            // r_valid[LATENCY - 2]  <= i_in_valid;
            // r_valid[LATENCY - 1]  <= r_valid[LATENCY - 2];
        end
    end

    initial begin
       $readmemh("stage3_fc1_bias.mem", bias_mem);
    end

    integer i;
    always @(*) begin
        for (i = 0;i<`acc_CO ;i = i + 1 ) begin
            w_ot_result[i] = 0;
            w_ot_result[i] = $signed(o_ot_ci_acc[i * `ACC_BW+:`ACC_BW]) + $signed(bias_mem[i]);
        end
    end

    integer j;
    always @(posedge clk, negedge reset_n) begin
        if (!reset_n) begin
            r_ot_result <= 0;
        end else if (i_in_valid) begin
            for (j = 0;j < `core_CO ; j = j + 1) begin
                r_ot_result[j * `OUT_BW +: `OUT_BW] <= $signed(w_ot_result[j]);
            end
        end
    end

    // reg signed [`OUT_BW -1:0] d_ot_result;
    // always @(posedge clk) begin
    //     if (r_valid[LATENCY - 1]) begin
    //             d_ot_result= r_ot_result[0 +: `OUT_BW];
    //         end
    // end

    reg signed [`OUT_BW -1:0] d_ot_result [0:`core_CO-1];
    integer ch;
    always @(posedge clk) begin
        if (r_valid[LATENCY - 1]) begin
            for (ch = 0; ch < `core_CO; ch = ch + 1) begin
                d_ot_result [ch] = r_ot_result[ch * `OUT_BW +: `OUT_BW];
            end
        end
    end
    
    assign o_ot_valid = r_valid[LATENCY - 1];
    assign o_ot_result = r_ot_result;

endmodule
