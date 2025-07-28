// DSP 최적화된 CI 채널 누산기
`timescale 1ns / 1ps
module cnn_acc_ci #(
    parameter I_F_BW = 8,
    parameter KX = 5,
    parameter KY = 5,
    parameter W_BW = 8,
    parameter CI = 1,
    parameter AK_BW = 21,
    parameter ACI_BW = 21
) (
    // Clock & Reset
    input                              clk,
    input                              reset_n,
    input signed [CI*KX*KY*W_BW-1:0]   i_cnn_weight,
    input                              i_in_valid,
    input        [KX*KY*I_F_BW-1:0]    i_window,
    output                             o_ot_valid,
    output signed [ACI_BW-1:0]         o_ot_ci_acc
);

    localparam LATENCY = 4;  // Increased for better pipelining

    //==============================================================================
    // Data Enable Signals 
    //==============================================================================
    reg [LATENCY-1:0] r_valid;
    wire [CI-1:0] w_kernel_valid;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[0] <= i_in_valid;
            r_valid[1] <= r_valid[0];
            r_valid[2] <= &w_kernel_valid;  // All kernels valid
            r_valid[3] <= r_valid[2];
        end
    end

    //==============================================================================
    // CNN Kernel Instances with DSP Optimization
    //==============================================================================
    wire signed [AK_BW-1:0] w_kernel_results [0:CI-1];

    genvar ci_idx;
    generate
        for (ci_idx = 0; ci_idx < CI; ci_idx = ci_idx + 1) begin : gen_ci_kernels
            wire [KX*KY*W_BW-1:0] w_cnn_weight = i_cnn_weight[ci_idx*KY*KX*W_BW +: KY*KX*W_BW];
            // Use the optimized CNN kernel
            cnn_kernel #(
                .KX(KX),
                .KY(KY),
                .I_F_BW(I_F_BW),
                .W_BW(W_BW),
                .M_BW(I_F_BW + W_BW),
                .AK_BW(AK_BW)
            ) U_cnn_kernel (
                .clk(clk),
                .reset_n(reset_n),
                .i_cnn_weight(w_cnn_weight),
                .i_in_valid(i_in_valid),
                .i_in_fmap(i_window),
                .o_ot_valid(w_kernel_valid[ci_idx]),
                .o_ot_kernel_acc(w_kernel_results[ci_idx])
            );
        end
    endgenerate

    //==============================================================================
    // Tree-based CI Accumulation for Better Timing
    //==============================================================================
    
    // Generate tree accumulation based on CI count
    generate
        if (CI == 1) begin : gen_ci_1
            reg signed [ACI_BW-1:0] r_ci_acc;
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_ci_acc <= {ACI_BW{1'b0}};
                end else if (r_valid[2]) begin
                    r_ci_acc <= {{(ACI_BW-AK_BW){w_kernel_results[0][AK_BW-1]}}, w_kernel_results[0]};
                end
            end
            assign o_ot_ci_acc = r_ci_acc;
            
        end else if (CI == 2) begin : gen_ci_2
            reg signed [ACI_BW-1:0] r_ci_acc;
            wire signed [ACI_BW-1:0] sum_temp = 
                {{(ACI_BW-AK_BW){w_kernel_results[0][AK_BW-1]}}, w_kernel_results[0]} +
                {{(ACI_BW-AK_BW){w_kernel_results[1][AK_BW-1]}}, w_kernel_results[1]};
                
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_ci_acc <= {ACI_BW{1'b0}};
                end else if (r_valid[2]) begin
                    r_ci_acc <= sum_temp;
                end
            end
            assign o_ot_ci_acc = r_ci_acc;
            
        end else if (CI == 4) begin : gen_ci_4
            // Level 1: 4->2
            reg signed [ACI_BW-1:0] level1_sum [0:1];
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    level1_sum[0] <= {ACI_BW{1'b0}};
                    level1_sum[1] <= {ACI_BW{1'b0}};
                end else if (r_valid[2]) begin
                    level1_sum[0] <= {{(ACI_BW-AK_BW){w_kernel_results[0][AK_BW-1]}}, w_kernel_results[0]} +
                                    {{(ACI_BW-AK_BW){w_kernel_results[1][AK_BW-1]}}, w_kernel_results[1]};
                    level1_sum[1] <= {{(ACI_BW-AK_BW){w_kernel_results[2][AK_BW-1]}}, w_kernel_results[2]} +
                                    {{(ACI_BW-AK_BW){w_kernel_results[3][AK_BW-1]}}, w_kernel_results[3]};
                end
            end
            
            // Level 2: 2->1
            reg signed [ACI_BW-1:0] r_ci_acc;
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_ci_acc <= {ACI_BW{1'b0}};
                end else if (r_valid[3]) begin
                    r_ci_acc <= level1_sum[0] + level1_sum[1];
                end
            end
            assign o_ot_ci_acc = r_ci_acc;
            
        end else begin : gen_ci_general
            // General case: Sequential accumulation with pipelining
            reg signed [ACI_BW-1:0] r_ci_acc;
            reg signed [ACI_BW-1:0] acc_temp;
            
            integer i;
            always @(*) begin
                acc_temp = {ACI_BW{1'b0}};
                for (i = 0; i < CI; i = i + 1) begin
                    acc_temp = acc_temp + {{(ACI_BW-AK_BW){w_kernel_results[i][AK_BW-1]}}, w_kernel_results[i]};
                end
            end
            
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_ci_acc <= {ACI_BW{1'b0}};
                end else if (r_valid[2]) begin
                    r_ci_acc <= acc_temp;
                end
            end
            assign o_ot_ci_acc = r_ci_acc;
        end
    endgenerate

    //==============================================================================
    // Output Assignment
    //==============================================================================
    assign o_ot_valid = r_valid[LATENCY-1];

endmodule
