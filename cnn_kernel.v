`timescale 1ns / 1ps
module cnn_kernel #(
    parameter KX = 5,          // Number of Kernel X
    parameter KY = 5,          // Number of Kernel Y
    parameter I_F_BW = 8,      // Bit Width of Input Feature
    parameter W_BW = 8,        // BW of weight parameter
    parameter B_BW = 16,       // BW of bias parameter
    parameter AK_BW = 21,      // M_BW + log(KY*KX) Accum Kernel 
    parameter M_BW = 16        // I_F_BW * W_BW
)(
    // Clock & Reset
    input clk,
    input reset_n,
    input [KX*KY*W_BW-1 : 0] i_cnn_weight,
    input i_in_valid,
    input [KX*KY*I_F_BW-1 : 0] i_in_fmap,
    output o_ot_valid,
    output signed [AK_BW-1 : 0] o_ot_kernel_acc
);

    localparam LATENCY = 3;
    localparam KERNEL_SIZE = KX * KY;

    //==============================================================================
    // Data Enable Signals 
    //==============================================================================
    reg [LATENCY-1 : 0] r_valid;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[0] <= i_in_valid;
            r_valid[1] <= r_valid[0];
            r_valid[2] <= r_valid[1];
        end
    end

    //==============================================================================
    // Input Data Pipelining
    //==============================================================================
    reg [I_F_BW-1:0] r_fmap [0:KERNEL_SIZE-1];
    reg [W_BW-1:0] r_weight [0:KERNEL_SIZE-1];

    integer i;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                r_fmap[i] <= {I_F_BW{1'b0}};
                r_weight[i] <= {W_BW{1'b0}};
            end
        end else if (i_in_valid) begin
            for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                r_fmap[i] <= i_in_fmap[i * I_F_BW +: I_F_BW];
                r_weight[i] <= i_cnn_weight[i * W_BW +: W_BW];
            end
        end
    end

    //==============================================================================
    // DSP Multiplier Instances
    //==============================================================================
    wire signed [M_BW-1:0] mul_results [0:KERNEL_SIZE-1];

    genvar mul_idx;
    generate
        for (mul_idx = 0; mul_idx < KERNEL_SIZE; mul_idx = mul_idx + 1) begin : gen_dsp_mul
            dsp_multiplier #(
                .A_BW(I_F_BW + 1),  // +1 for sign extension
                .B_BW(W_BW),
                .P_BW(M_BW)
            ) U_dsp_mul (
                .clk(clk),
                .reset_n(reset_n),
                .enable(r_valid[0]),
                .a({1'b0, r_fmap[mul_idx]}),      // Zero-extend unsigned 8bit to signed 9bit
                .b($signed(r_weight[mul_idx])),   // Signed weight
                .product(mul_results[mul_idx])
            );
        end
    endgenerate

    //==============================================================================
    // Accumulation using Tree Structure for Better Timing
    //==============================================================================
    reg signed [AK_BW-1:0] acc_temp;
    reg signed [AK_BW-1:0] r_acc_kernel;

    // Tree-based accumulation for better timing
    always @(*) begin
        acc_temp = {AK_BW{1'b0}};
        for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
            acc_temp = acc_temp + {{(AK_BW-M_BW){mul_results[i][M_BW-1]}}, mul_results[i]};
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_acc_kernel <= {AK_BW{1'b0}};
        end else if (r_valid[1]) begin
            r_acc_kernel <= acc_temp;
        end
    end

    //==============================================================================
    // Optional: 2D Array Access for Debug (if needed)
    //==============================================================================
    reg signed [M_BW-1:0] debug_mul_2d [0:KY-1][0:KX-1];
    reg [I_F_BW-1:0] debug_fmap_2d [0:KY-1][0:KX-1];
    reg [W_BW-1:0] debug_weight_2d [0:KY-1][0:KX-1];

    integer k, j;
    always @(posedge clk) begin
        for (k = 0; k < KY; k = k + 1) begin
            for (j = 0; j < KX; j = j + 1) begin
                debug_mul_2d[k][j] <= mul_results[k*KX + j];
                debug_fmap_2d[k][j] <= r_fmap[k*KX + j];
                debug_weight_2d[k][j] <= r_weight[k*KX + j];
            end
        end
    end

    //==============================================================================
    // Output Assignment
    //==============================================================================
    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_kernel_acc = r_acc_kernel;

endmodule