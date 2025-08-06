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
`include "defines_cnn_core.v"

module stage3_cnn_core(
    input clk,
    input reset_n,

    // pooling valid
    input i_in_valid,
    // 48 개 중 0, 16, 32
    input [`acc_CO * `ST3_ACC_BW-1:0] o_ot_ci_acc,
    
    output o_ot_valid,
    //output [`core_CO * `ST3_OUT_BW -1:0] o_ot_result,

    output [`ST3_OUT_BW-1:0] o_cnn_value,
    output [$clog2(`acc_CO)-1:0] index_info
    );
    // bias
    localparam LATENCY = 1;
    reg signed [`ST3_BIAS_BW-1:0] bias_mem[0:`acc_CO-1];
    

    //reg signed [`CO * `ST3_OUT_BW -1:0] w_ot_result;
    reg signed [`ST3_OUT_BW -1:0] w_ot_result[0:`core_CO-1];
    // reg signed [`ST3_OUT_BW -1:0] w_ot_result1;
    // reg signed [`ST3_OUT_BW -1:0] w_ot_result2;

    reg signed [`core_CO * `ST3_OUT_BW -1:0] r_ot_result;

    // (* mark_debug = "true" *) reg signed [`ST3_OUT_BW -1:0] d_ot_result0;
    // (* mark_debug = "true" *) reg signed [`ST3_OUT_BW -1:0] d_ot_result1;
    // (* mark_debug = "true" *) reg signed [`ST3_OUT_BW -1:0] d_ot_result2;
    // always @(posedge clk, negedge reset_n) begin
    //     if (!reset_n) begin
    //         d_ot_result0 <= 0;
    //         d_ot_result1 <= 0;
    //         d_ot_result2 <= 0;
    //     end else begin
    //         d_ot_result0 <= r_ot_result[0+:`ST3_OUT_BW];
    //         d_ot_result1 <= r_ot_result[`ST3_OUT_BW+:`ST3_OUT_BW];
    //         d_ot_result2 <= r_ot_result[2*`ST3_OUT_BW+:`ST3_OUT_BW];
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
            w_ot_result[i] = $signed(o_ot_ci_acc[i * `ST3_ACC_BW+:`ST3_ACC_BW]) + $signed(bias_mem[i]);
        end
    end

    integer j;
    reg result_valid;
    always @(posedge clk, negedge reset_n) begin
        if (!reset_n) begin
            r_ot_result <= 0;
        end else if (i_in_valid) begin
            for (j = 0;j < `core_CO ; j = j + 1) begin
                r_ot_result[j * `ST3_OUT_BW +: `ST3_OUT_BW] <= $signed(w_ot_result[j]);
            end
        end
    end
    
    reg [$clog2(`core_CO)-1:0]addr_cnt_reg, addr_cnt_next;

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            result_valid <=0;
        end else if (i_in_valid)begin
            result_valid <=1;
        end else if(addr_cnt_reg == `acc_CO-1)begin
            result_valid <=0;
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            addr_cnt_reg <=0;
        end else begin
            addr_cnt_reg <= addr_cnt_next;
        end
    end

    always @(*) begin
        addr_cnt_next = addr_cnt_reg;
        if(result_valid) begin
            if(addr_cnt_reg == `acc_CO-1) begin
                addr_cnt_next =0;
            end else begin
                addr_cnt_next = addr_cnt_reg +1;            
            end
        end
    end

    assign o_cnn_value = r_ot_result[addr_cnt_reg*`ST3_OUT_BW+:`ST3_OF_BW];
    // reg signed [`ST3_OUT_BW -1:0] d_ot_result;
    // always @(posedge clk) begin
    //     if (r_valid[LATENCY - 1]) begin
    //             d_ot_result= r_ot_result[0 +: `ST3_OUT_BW];
    //         end
    // end

    //debug
    reg signed [`ST3_OUT_BW -1:0] d_ot_result [0:`core_CO-1];
    integer ch;
    always @(posedge clk) begin
        if (i_in_valid) begin
            for (ch = 0; ch < `core_CO; ch = ch + 1) begin
                d_ot_result [ch] <= $signed(w_ot_result[ch]);
            end
        end
    end
    
    assign o_ot_valid = result_valid;
    assign o_ot_result = r_ot_result;
    assign index_info = addr_cnt_reg;

endmodule
