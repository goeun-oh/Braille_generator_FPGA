//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/20 02:37:58
// Design Name: 
// Module Name: top_cnn
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
`include "timescale.vh"
`include "defines_cnn_core.vh"

module top_cnn (
    input  wire                                                 clk,
    input  wire                                                 reset_n,
    input  wire                                                 i_in_valid,
    input  wire [`CI*`POOL_IN_SIZE*`POOL_IN_SIZE*`IF_BW-1:0]    i_in_fmap,
    output wire                                                 o_ot_valid,
    output wire [`CO*`OUT_BW-1:0]                               o_ot_bias
);

    wire                                               w_pool_matrix_valid;
    wire [`CI*`P_SIZE*`P_SIZE*`OF_BW-1:0]              w_pool_matrix_acc;

max_pool U_max_pool(
    .clk(clk),
    .reset_n(reset_n),
    .i_in_valid(i_in_valid),
    .i_in_fmap(i_in_fmap),
    .o_ot_valid(w_pool_matrix_valid),
    .o_ot_ci_acc(w_pool_matrix_acc)
);
matrixmultiplex U_matrixmultiplex(
    .clk(clk),
    .reset_n(reset_n),
    .i_in_valid(w_pool_matrix_valid),
    .i_in_fmap(w_pool_matrix_acc),
    .o_ot_valid(o_ot_valid),
    .o_ot_bias(o_ot_bias)
);

endmodule
