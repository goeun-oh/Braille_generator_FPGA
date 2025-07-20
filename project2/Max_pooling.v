`include "timescale.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/19 22:25:53
// Design Name: 
// Module Name: Max_pooling
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

`include "defines_cnn_core.vh"

module max_pool (
    clk,
    reset_n,
    i_in_valid,
    i_in_fmap,
    o_ot_valid,
    o_ot_ci_acc
);

    input                       clk;
    input                       reset_n;
    input                       i_in_valid;
    input  [`CI*8*8*32-1:0]     i_in_fmap;
    output                      o_ot_valid;
    output [`CI*4*4*32-1:0]     o_ot_ci_acc;

    // Valid 신호 처리
    reg r_valid;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid <= 1'b0;
        end else begin
            r_valid <= i_in_valid;
        end
    end

    // Max pooling 결과 저장
    reg [`CI*4*4*32-1:0] r_pool;
    wire [`CI*4*4*32-1:0] w_pool;

    // Max pooling 연산
    genvar ci_max_pool, row_max_pool, col_max_pool;
    generate
        for(ci_max_pool = 0; ci_max_pool < 3; ci_max_pool = ci_max_pool + 1) begin : gen_channel
            for(row_max_pool = 0; row_max_pool < 4; row_max_pool = row_max_pool + 1) begin : gen_row
                for(col_max_pool = 0; col_max_pool < 4; col_max_pool = col_max_pool + 1) begin : gen_col
                    max_pool_2x2 U_max_pool_2x2(
                        .i00(i_in_fmap[32*((64*ci_max_pool)+16*row_max_pool+2*col_max_pool)+:32]),
                        .i01(i_in_fmap[32*((64*ci_max_pool)+16*row_max_pool+2*col_max_pool+1)+:32]),
                        .i10(i_in_fmap[32*((64*ci_max_pool)+16*row_max_pool+8+2*col_max_pool)+:32]),
                        .i11(i_in_fmap[32*((64*ci_max_pool)+16*row_max_pool+8+2*col_max_pool+1)+:32]),
                        .o_max(w_pool[32*((16*ci_max_pool)+4*row_max_pool+col_max_pool)+:32])
                    );
                end
            end
        end
    endgenerate

    // 출력 레지스터
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_pool <= {`CI*4*4*32{1'b0}};
        end else if(i_in_valid) begin
            r_pool <= w_pool;
        end
    end

    assign o_ot_valid = r_valid;
    assign o_ot_ci_acc = r_pool;

endmodule

// 32는 1 pixel 당 bit 수
module max_pool_2x2 (
    input  [32-1:0] i00, i01, i10, i11,
    output [32-1:0] o_max
);
    wire [32-1:0] max0 = (i00 > i01) ? i00 : i01;
    wire [32-1:0] max1 = (i10 > i11) ? i10 : i11;
    assign o_max = (max0 > max1) ? max0 : max1;
endmodule

