`include "timescale.vh"
`include "defines_cnn_core.vh"

module max_pool (
    input  wire                                               clk,
    input  wire                                               reset_n,
    input  wire                                               i_in_valid,
    input  wire [`CI*`POOL_IN_SIZE*`POOL_IN_SIZE*`IF_BW-1:0]  i_in_fmap,
    output wire                                               o_ot_valid,
    output wire [`CI*`P_SIZE*`P_SIZE*`OF_BW-1:0]              o_ot_ci_acc
);

    // 1-cycle latency valid signal
    reg r_valid;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) r_valid <= 1'b0;
        else          r_valid <= i_in_valid;
    end

    wire [`CI*`P_SIZE*`P_SIZE*`OF_BW-1:0] w_pool;
    reg  [`CI*`P_SIZE*`P_SIZE*`OF_BW-1:0] r_pool;

    // Parameterized Max-Pooling Operation
    genvar ci, row, col;
    generate
        for (ci = 0; ci < `CI; ci = ci + 1) begin : GEN_CH
            for (row = 0; row < `P_SIZE; row = row + 1) begin : GEN_ROW
                for (col = 0; col < `P_SIZE; col = col + 1) begin : GEN_COL
                    localparam r_orig = row * `POOL_K;
                    localparam c_orig = col * `POOL_K;

                    max_pool_2x2 U_max_pool (
                        .i00(i_in_fmap[`IF_BW*((ci*`POOL_IN_SIZE + r_orig+0)*`POOL_IN_SIZE + c_orig+0) +: `IF_BW]),
                        .i01(i_in_fmap[`IF_BW*((ci*`POOL_IN_SIZE + r_orig+0)*`POOL_IN_SIZE + c_orig+1) +: `IF_BW]),
                        .i10(i_in_fmap[`IF_BW*((ci*`POOL_IN_SIZE + r_orig+1)*`POOL_IN_SIZE + c_orig+0) +: `IF_BW]),
                        .i11(i_in_fmap[`IF_BW*((ci*`POOL_IN_SIZE + r_orig+1)*`POOL_IN_SIZE + c_orig+1) +: `IF_BW]),
                        .o_max(w_pool[`OF_BW*((ci*`P_SIZE + row)*`P_SIZE + col) +: `OF_BW])
                    );
                end
            end
        end
    endgenerate

    // Output Register
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n)        r_pool <= {`CI*`P_SIZE*`P_SIZE*`OF_BW{1'b0}};
        else if (i_in_valid) r_pool <= w_pool;
    end

    assign o_ot_valid  = r_valid;
    assign o_ot_ci_acc = r_pool;

endmodule

// Sub-module
module max_pool_2x2 (
    input  [`OF_BW-1:0] i00, i01, i10, i11,
    output [`OF_BW-1:0] o_max
);
    wire [`OF_BW-1:0] max0 = ($signed(i00) > $signed(i01)) ? i00 : i01;
    wire [`OF_BW-1:0] max1 = ($signed(i10) > $signed(i11)) ? i10 : i11;
    assign o_max = ($signed(max0) > $signed(max1)) ? max0 : max1;
endmodule


