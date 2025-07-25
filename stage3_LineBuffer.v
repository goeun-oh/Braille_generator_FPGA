 `timescale 1ns / 1ps
// //////////////////////////////////////////////////////////////////////////////////
// // Company: 
// // Engineer: 
// // 
// // Create Date: 2025/07/21 16:09:05
// // Design Name: 
// // Module Name: LineBuffer
// // Project Name: 
// // Target Devices: 
// // Tool Versions: 
// // Description: 
// // 
// // Dependencies: 
// // 
// // Revision:
// // Revision 0.01 - File Created
// // Additional Comments:
// // 
// //////////////////////////////////////////////////////////////////////////////////
// // 현재 목표 4라인 buffer 구현
 `include "stage3_defines_cnn_core.vh"

module line_buffer (
    input clk,
    input reset_n,

    input              i_in_valid,
    input [`IF_BW-1:0] i_in_pixel,

    output                              o_window_valid,
    output [`POOL_K*`POOL_K*`IF_BW-1:0] o_window
);

    // parameter LATENCY = 2;

    reg [$clog2(`POOL_IN_SIZE)-1:0] x_cnt, x_cnt_d;
    reg [$clog2(`POOL_IN_SIZE)-1:0] y_cnt, y_cnt_d;


    reg [`IF_BW-1:0] line_buf[0:`POOL_K-1][0:`POOL_IN_SIZE-1];  // 3줄만 저장. 최신 줄은 현재 pixel로 채움
    // 32 bit data 3행 8열

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_cnt <= 0;
            y_cnt <= 0;
        end else if (i_in_valid) begin
            if (x_cnt == `POOL_IN_SIZE - 1) begin
                x_cnt <= 0;
                if (y_cnt == `POOL_IN_SIZE - 1) begin
                    y_cnt <= 0;
                end else begin
                    y_cnt <= y_cnt + 1;
                end
            end else begin
                x_cnt <= x_cnt + 1;
            end
        end
    end

    // == 1클럭 딜레이 좌표 ==
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_cnt_d <= 0;
            y_cnt_d <= 0;
        end else begin
            x_cnt_d <= x_cnt;
            y_cnt_d <= y_cnt;
        end
    end

    always @(posedge clk) begin
        if (i_in_valid) begin
            line_buf[0][x_cnt] <= line_buf[1][x_cnt]; // 2칸전 <= 1칸전
            line_buf[1][x_cnt] <= i_in_pixel;  // 2칸전 <= 1칸전
            // line_buf[1][x_cnt] <= line_buf[2][x_cnt]; // 1칸전 <= 최신
            // line_buf[2][x_cnt] <= i_in_pixel;         // 최신 <= 새로운 입력
        end
    end

    reg [`POOL_K*`POOL_K*`IF_BW-1:0] r_window;
    //디버깅
    reg [`IF_BW-1:0] r_o_window [0:`POOL_K-1][0:`POOL_K-1];


    integer wy, wx;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_window <= 0;
        end else begin
            if (x_cnt_d >= `POOL_K - 1 && y_cnt_d >= `POOL_K-1) begin
                for (wy = 0; wy < `POOL_K; wy = wy + 1) begin
                    for (wx = 0; wx < `POOL_K; wx = wx + 1) begin
                        r_window[(wy*`POOL_K + wx)*`IF_BW +: `IF_BW] <=line_buf[wy][x_cnt_d-(`POOL_K-1)+wx];
                        //디버깅
                        r_o_window [wy][wx] <= line_buf[wy][x_cnt_d-(`POOL_K-1)+wx];
                    end
                end
            end
        end
    end
    reg r_window_valid;
    reg w_in_valid;
    always @(posedge clk, negedge reset_n) begin
        if(!reset_n) begin
            w_in_valid <=0;
        end else begin
            w_in_valid <= i_in_valid;
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_window_valid <= 0;
        end else if (i_in_valid
            && x_cnt_d >= (`POOL_K-1) && y_cnt_d >= (`POOL_K-1)
            && (((x_cnt_d - (`POOL_K-1)) % `STRIDE) == 0)
            && (((y_cnt_d - (`POOL_K-1)) % `STRIDE) == 0)
        ) begin
            r_window_valid <= 1;
        end else if (w_in_valid
            && x_cnt_d >= (`POOL_K-1) && y_cnt_d >= (`POOL_K-1)
            && (((x_cnt_d - (`POOL_K-1)) % `STRIDE) == 0)
            && (((y_cnt_d - (`POOL_K-1)) % `STRIDE) == 0)
        ) begin
            r_window_valid <= 1;
        end else begin
            r_window_valid <= 0;
        end
    end

    assign o_window = r_window;
    assign o_window_valid = r_window_valid;


endmodule
