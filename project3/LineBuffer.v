`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/21 16:09:05
// Design Name: 
// Module Name: LineBuffer
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
// 현재 목표 4라인 buffer 구현
`include "defines_cnn_core.vh"

module line_buffer (
    input clk,
    input reset_n,

    input i_in_valid,
    input [`IF_BW-1:0] i_in_pixel,
    
    output o_window_valid,
    output [`POOL_K*`POOL_K*`IF_BW-1:0] o_window
);

    // parameter LATENCY = 2;

    reg [$clog2(`POOL_IN_SIZE)-1:0] x_cnt;
    reg [$clog2(`POOL_IN_SIZE)-1:0] y_cnt;


    reg [`IF_BW-1:0] line_buf[0:`POOL_K][0:`POOL_IN_SIZE-1];  // 3줄만 저장. 최신 줄은 현재 pixel로 채움
    // 32 bit data 3행 8열

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_cnt <=0;
            y_cnt <=0;
        end else if (i_in_valid) begin
            
            if (x_cnt == `POOL_IN_SIZE-1) begin
                x_cnt <=0;
                if (y_cnt == `POOL_IN_SIZE-1) begin
                    y_cnt <= y_cnt + 1;
                end
            end else begin
                x_cnt <= x_cnt + 1;
            end
        end
    end
    
    integer i;
    always @(posedge clk) begin
        if(i_in_valid) begin
            for (i=0; i < `POOL_K; i = i+1) begin
                line_buf[i][x_cnt] <= line_buf[i+1][x_cnt];
            end
            line_buf[`POOL_K][x_cnt] <= i_in_pixel;
        end
    end

    reg [`POOL_K*`POOL_K*`IF_BW-1:0] r_window;   

    //디버깅

    integer wy, wx;
    always @(posedge clk, posedge reset_n) begin
        if(!reset_n) begin
            for (wy =0; wy <`POOL_K; wy = wy + 1) begin
                for (wx =0; wx <`POOL_K; wx = wx +1) begin
                    if(x_cnt >= `POOL_K-1 && y_cnt >= `POOL_K-1) begin
                        r_window[(wy*`POOL_K + wx)*`IF_BW +: `IF_BW] = 0;
                    end
                end
            end    
        end else begin
            for (wy =0; wy <`POOL_K; wy = wy + 1) begin
                for (wx =0; wx <`POOL_K; wx = wx +1) begin
                    if(x_cnt >= `POOL_K-1 && y_cnt >= `POOL_K-1) begin
                        r_window[(wy*`POOL_K + wx)*`IF_BW +: `IF_BW] =line_buf[wy][x_cnt-(`POOL_K-1- wx)];
                    end
                end
            end 
        end
    end

    reg r_window_valid;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_window_valid <= 0;
        end else if (i_in_valid
            && x_cnt >= (`POOL_K-1) && y_cnt >= (`POOL_K-1)
            && (((x_cnt - (`POOL_K-1)) % `STRIDE) == 0)
            && (((y_cnt - (`POOL_K-1)) % `STRIDE) == 0)
        ) begin
            r_window_valid <= 1;
        end else begin
            r_window_valid <= 0;
        end
    end

    assign o_window = r_window;
    assign o_window_valid = r_window_valid;


endmodule