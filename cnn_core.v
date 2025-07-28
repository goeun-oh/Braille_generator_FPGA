// DSP 최적화된 CNN 코어 모듈
`timescale 1ns / 1ps
module cnn_core #(
    parameter I_F_BW = 8,
    parameter KX = 5,
    parameter KY = 5,
    parameter IX = 28,
    parameter W_BW = 8,
    parameter B_BW = 16,  // bias
    parameter CI = 1,
    parameter CO = 3,
    parameter AK_BW = 21,
    parameter ACI_BW = 21,
    parameter O_F_BW = 20,
    parameter AB_BW = 21,
    parameter AR_BW = 20
) (
    // Clock & Reset
    input                           clk,
    input                           reset_n,
    input [CO*CI*KX*KY*W_BW-1 : 0] i_cnn_weight,
    input [         CO*B_BW-1 : 0] i_cnn_bias,
    input                           i_in_valid,
    input  [       CI*I_F_BW-1 : 0] i_in_fmap,    // CI 채널 입력으로 수정
    output                          o_ot_valid,
    output [       CO*O_F_BW-1 : 0] o_ot_fmap
);

    localparam LATENCY = 6;  // 파이프라인 증가 (line_buffer + acc_ci + bias + activation)

    //==============================================================================
    // Data Enable Signals 
    //==============================================================================
    reg [LATENCY-1:0] r_valid;
    wire [CO-1:0] w_acc_valid;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[0] <= i_in_valid;
            r_valid[1] <= r_valid[0];
            r_valid[2] <= r_valid[1];
            r_valid[3] <= r_valid[2];
            r_valid[4] <= &w_acc_valid;  // All CO channels valid
            r_valid[5] <= r_valid[4];
        end
    end

    //==============================================================================
    // Line Buffer Instances (CI 채널별)
    //==============================================================================
    wire [KX*KY*I_F_BW-1:0] w_windows [0:CI-1];
    wire [CI-1:0] w_window_valid;

    genvar ci_idx;
    generate
        for (ci_idx = 0; ci_idx < CI; ci_idx = ci_idx + 1) begin : gen_line_buffers
            line_buffer #(
                .KX(KX), 
                .KY(KY), 
                .I_F_BW(I_F_BW)
            ) U_line_buffer (
                .clk(clk),
                .reset_n(reset_n),
                .i_in_valid(i_in_valid),
                .i_in_pixel(i_in_fmap[ci_idx*I_F_BW +: I_F_BW]),  // 각 채널별 입력
                .o_window_valid(w_window_valid[ci_idx]),
                .o_window(w_windows[ci_idx])
            );
        end
    endgenerate

    // 모든 채널의 window가 valid할 때만 처리
    wire all_windows_valid = &w_window_valid;

    //==============================================================================
    // CI 채널 데이터 결합 (인터리빙)
    //==============================================================================
    reg [CI*KX*KY*I_F_BW-1:0] combined_windows;
    integer pos_idx, ci_idx_comb, spatial_idx;
    
    always @(*) begin
        for (spatial_idx = 0; spatial_idx < KX*KY; spatial_idx = spatial_idx + 1) begin
            for (ci_idx_comb = 0; ci_idx_comb < CI; ci_idx_comb = ci_idx_comb + 1) begin
                pos_idx = spatial_idx * CI + ci_idx_comb;
                combined_windows[pos_idx*I_F_BW +: I_F_BW] = 
                    w_windows[ci_idx_comb][spatial_idx*I_F_BW +: I_F_BW];
            end
        end
    end

    //==============================================================================
    // CNN Accumulator CI Instances (CO 출력별)
    //==============================================================================
    wire signed [ACI_BW-1:0] w_acc_results [0:CO-1];

    genvar co_idx;
    generate
        for (co_idx = 0; co_idx < CO; co_idx = co_idx + 1) begin : gen_co_acc
            wire signed [CI*KX*KY*W_BW-1:0] w_co_weights = 
                i_cnn_weight[co_idx*CI*KX*KY*W_BW +: CI*KX*KY*W_BW];

            cnn_acc_ci #(
                .I_F_BW(I_F_BW),
                .KX(KX), .KY(KY),
                .W_BW(W_BW),
                .CI(CI),
                .AK_BW(AK_BW),
                .ACI_BW(ACI_BW)
            ) U_cnn_acc_ci (
                .clk(clk),
                .reset_n(reset_n),
                .i_cnn_weight(w_co_weights),
                .i_in_valid(all_windows_valid),
                .i_window(combined_windows),
                .o_ot_valid(w_acc_valid[co_idx]),
                .o_ot_ci_acc(w_acc_results[co_idx])
            );
        end
    endgenerate

    //==============================================================================
    // Bias Addition with DSP Optimization
    //==============================================================================
    reg signed [AB_BW-1:0] stage_bias [0:CO-1];
    
    genvar bias_idx;
    generate
        for (bias_idx = 0; bias_idx < CO; bias_idx = bias_idx + 1) begin : gen_bias_add
            wire signed [B_BW-1:0] bias_val = $signed(i_cnn_bias[bias_idx*B_BW +: B_BW]);
            
            (* use_dsp = "yes" *)
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    stage_bias[bias_idx] <= {AB_BW{1'b0}};
                end else if (r_valid[4]) begin
                    // Sign extend and add bias
                    stage_bias[bias_idx] <= 
                        {{(AB_BW-ACI_BW){w_acc_results[bias_idx][ACI_BW-1]}}, w_acc_results[bias_idx]} + 
                        {{(AB_BW-B_BW){bias_val[B_BW-1]}}, bias_val};
                end
            end
        end
    endgenerate

    //==============================================================================
    // ReLU Activation (Optimized)
    //==============================================================================
    reg signed [AR_BW-1:0] stage_relu [0:CO-1];
    
    genvar relu_idx;
    generate
        for (relu_idx = 0; relu_idx < CO; relu_idx = relu_idx + 1) begin : gen_relu
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    stage_relu[relu_idx] <= {AR_BW{1'b0}};
                end else if (r_valid[5]) begin
                    // ReLU: max(0, x)
                    if (stage_bias[relu_idx][AB_BW-1] == 1'b1) begin  // Negative
                        stage_relu[relu_idx] <= {AR_BW{1'b0}};
                    end else begin  // Positive
                        // Saturation handling
                        if (stage_bias[relu_idx] > ((1 << AR_BW) - 1)) begin
                            stage_relu[relu_idx] <= (1 << AR_BW) - 1;  // Saturate to max
                        end else begin
                            stage_relu[relu_idx] <= stage_bias[relu_idx][AR_BW-1:0];
                        end
                    end
                end
            end
        end
    endgenerate

    //==============================================================================
    // Output Packing
    //==============================================================================
    reg [CO*O_F_BW-1:0] r_output;
    
    genvar out_idx;
    generate
        for (out_idx = 0; out_idx < CO; out_idx = out_idx + 1) begin : gen_output
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_output[out_idx*O_F_BW +: O_F_BW] <= {O_F_BW{1'b0}};
                end else if (r_valid[5]) begin
                    // Truncate or extend to output bit width
                    if (AR_BW >= O_F_BW) begin
                        r_output[out_idx*O_F_BW +: O_F_BW] <= stage_relu[out_idx][O_F_BW-1:0];
                    end else begin
                        r_output[out_idx*O_F_BW +: O_F_BW] <= {{(O_F_BW-AR_BW){1'b0}}, stage_relu[out_idx]};
                    end
                end
            end
        end
    endgenerate

    //==============================================================================
    // Debug Signals (Optional)
    //==============================================================================
    // 디버깅용 2D 배열 (필요시 사용)
    reg signed [ACI_BW-1:0] debug_acc_2d [0:CO-1];
    reg signed [AB_BW-1:0] debug_bias_2d [0:CO-1];
    reg signed [AR_BW-1:0] debug_relu_2d [0:CO-1];
    
    integer debug_idx;
    always @(posedge clk) begin
        for (debug_idx = 0; debug_idx < CO; debug_idx = debug_idx + 1) begin
            debug_acc_2d[debug_idx] <= w_acc_results[debug_idx];
            debug_bias_2d[debug_idx] <= stage_bias[debug_idx];
            debug_relu_2d[debug_idx] <= stage_relu[debug_idx];
        end
    end

    //==============================================================================
    // Output Assignment
    //==============================================================================
    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_fmap = r_output;

endmodule
