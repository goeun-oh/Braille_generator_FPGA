// DSP 최적화된 Stage2 CNN CI 누산기
`timescale 1ns / 1ps
`include "stage2_defines_cnn_core.v"

module stage2_cnn_acc_ci (
    // Clock & Reset
    input                                           		clk,
    input                                           		reset_n,

    // 3*5*5*8 = 600bit weights
    input     signed [`ST2_Conv_CI*`KX*`KY*`W_BW-1 : 0]  			i_cnn_weight,
    input                                           				i_in_valid,
    // 3*5*5*20 = 1500bit feature maps  
    input     signed [`ST2_Conv_CI*`KX*`KY*`ST2_Conv_IBW-1 : 0]  	i_in_fmap,
    output                                          				o_ot_valid,
    output    signed [`ACI_BW-1 : 0]  			            		o_ot_ci_acc 	     
);

    localparam LATENCY = 7;  // 증가된 파이프라인 (커널 5 + CI 누산 2)
    localparam CI = `ST2_Conv_CI;  // 3

    //==============================================================================
    // Pipeline Control
    //==============================================================================
    reg [LATENCY-1:0] r_valid;
    wire [CI-1:0] w_kernel_valid;

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[0] <= i_in_valid;
            r_valid[1] <= r_valid[0];
            r_valid[2] <= r_valid[1];
            r_valid[3] <= r_valid[2];
            r_valid[4] <= r_valid[3];
            r_valid[5] <= &w_kernel_valid;  // All CI kernels valid
            r_valid[6] <= r_valid[5];
        end
    end

    //==============================================================================
    // CI Channel Kernel Instances (DSP 최적화 버전 사용)
    //==============================================================================
    wire signed [`AK_BW-1:0] w_kernel_results [0:CI-1];

    genvar ci_idx;
    generate
        for(ci_idx = 0; ci_idx < CI; ci_idx = ci_idx + 1) begin : gen_ci_kernels
            // 각 CI 채널별 weight와 feature map 추출
            wire signed [`KX*`KY*`W_BW-1:0] ci_weight = 
                i_cnn_weight[ci_idx*`KY*`KX*`W_BW +: `KY*`KX*`W_BW];
            wire signed [`KX*`KY*`ST2_Conv_IBW-1:0] ci_fmap = 
                i_in_fmap[ci_idx*`KY*`KX*`ST2_Conv_IBW +: `KY*`KX*`ST2_Conv_IBW];

            // DSP 최적화된 Stage2 커널 사용
            stage2_cnn_kernel U_stage2_kernel (
                .clk(clk),
                .reset_n(reset_n),
                .i_cnn_weight(ci_weight),
                .i_in_valid(i_in_valid),
                .i_in_fmap(ci_fmap),
                .o_ot_valid(w_kernel_valid[ci_idx]),
                .o_ot_kernel_acc(w_kernel_results[ci_idx])
            );
        end
    endgenerate

    //==============================================================================
    // CI 채널 누산 최적화 (CI=3 전용)
    //==============================================================================
    
    generate
        if (CI == 1) begin : gen_ci_single
            // CI=1인 경우: 단순 할당
            reg signed [`ACI_BW-1:0] r_single_result;
            
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_single_result <= {`ACI_BW{1'b0}};
                end else if (w_kernel_valid[0]) begin
                    r_single_result <= {{(`ACI_BW-`AK_BW){w_kernel_results[0][`AK_BW-1]}}, w_kernel_results[0]};
                end
            end
            
            assign o_ot_ci_acc = r_single_result;
            
        end else if (CI == 2) begin : gen_ci_dual
            // CI=2인 경우: 단일 덧셈
            reg signed [`ACI_BW-1:0] r_dual_result;
            
            (* use_dsp = "yes" *)  // DSP로 덧셈 최적화
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_dual_result <= {`ACI_BW{1'b0}};
                end else if (&w_kernel_valid) begin
                    r_dual_result <= 
                        {{(`ACI_BW-`AK_BW){w_kernel_results[0][`AK_BW-1]}}, w_kernel_results[0]} +
                        {{(`ACI_BW-`AK_BW){w_kernel_results[1][`AK_BW-1]}}, w_kernel_results[1]};
                end
            end
            
            assign o_ot_ci_acc = r_dual_result;
            
        end else if (CI == 3) begin : gen_ci_triple
            // CI=3인 경우 (Stage2 기본): 2단계 파이프라인 덧셈
            reg signed [`ACI_BW-1:0] stage1_partial_sum;
            reg signed [`ACI_BW-1:0] stage2_final_sum;
            
            // Stage 1: 처음 두 개 더하기
            (* use_dsp = "yes" *)
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    stage1_partial_sum <= {`ACI_BW{1'b0}};
                end else if (&w_kernel_valid) begin
                    stage1_partial_sum <= 
                        {{(`ACI_BW-`AK_BW){w_kernel_results[0][`AK_BW-1]}}, w_kernel_results[0]} +
                        {{(`ACI_BW-`AK_BW){w_kernel_results[1][`AK_BW-1]}}, w_kernel_results[1]};
                end
            end
            
            // Stage 2: 세 번째 더하기
            (* use_dsp = "yes" *)
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    stage2_final_sum <= {`ACI_BW{1'b0}};
                end else if (r_valid[5]) begin
                    stage2_final_sum <= stage1_partial_sum + 
                        {{(`ACI_BW-`AK_BW){w_kernel_results[2][`AK_BW-1]}}, w_kernel_results[2]};
                end
            end
            
            assign o_ot_ci_acc = stage2_final_sum;
            
        end else begin : gen_ci_general
            // CI > 3인 경우: 일반적인 트리 구조
            reg signed [`ACI_BW-1:0] r_general_result;
            reg signed [`ACI_BW-1:0] acc_temp;
            
            integer i;
            always @(*) begin
                acc_temp = {`ACI_BW{1'b0}};
                for (i = 0; i < CI; i = i + 1) begin
                    acc_temp = acc_temp + {{(`ACI_BW-`AK_BW){w_kernel_results[i][`AK_BW-1]}}, w_kernel_results[i]};
                end
            end
            
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_general_result <= {`ACI_BW{1'b0}};
                end else if (&w_kernel_valid) begin
                    r_general_result <= acc_temp;
                end
            end
            
            assign o_ot_ci_acc = r_general_result;
        end
    endgenerate

    //==============================================================================
    // Output Assignment
    //==============================================================================
    assign o_ot_valid = r_valid[LATENCY-1];

endmodule
