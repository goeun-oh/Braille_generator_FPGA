// DSP 최적화된 Stage2 CNN 커널
`timescale 1ns / 1ps
`include "stage2_defines_cnn_core.v"

module stage2_cnn_kernel (
    // Clock & Reset
    input                               		   clk         	,
    input                               		   reset_n     	,

    //5x5x7
    input     signed [`KX*`KY*`W_BW-1 : 0] 	       i_cnn_weight ,
    input                                          i_in_valid  	,
    input     signed [`KX*`KY*`ST2_Conv_IBW-1 : 0] i_in_fmap    , //5x5x(20bit)
    output                                         o_ot_valid  	,
    output    signed [`AK_BW-1 : 0]  			   o_ot_kernel_acc           
);

    localparam LATENCY = 5;  // 증가된 파이프라인 깊이
    localparam KERNEL_SIZE = `KX * `KY;  // 5x5 = 25

    //==============================================================================
    // Data Enable Signals 
    //==============================================================================
    reg [LATENCY-1:0] r_valid;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[0] <= i_in_valid;
            r_valid[1] <= r_valid[0];
            r_valid[2] <= r_valid[1];
            r_valid[3] <= r_valid[2];
            r_valid[4] <= r_valid[3];
        end
    end

    //==============================================================================
    // Input Data Pipeline Stage 1
    //==============================================================================
    reg signed [`ST2_Conv_IBW-1:0] stage1_fmap [0:KERNEL_SIZE-1];
    reg signed [`W_BW-1:0] stage1_weight [0:KERNEL_SIZE-1];

    integer i;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                stage1_fmap[i] <= {`ST2_Conv_IBW{1'b0}};
                stage1_weight[i] <= {`W_BW{1'b0}};
            end
        end else if (i_in_valid) begin
            for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                stage1_fmap[i] <= i_in_fmap[i * `ST2_Conv_IBW +: `ST2_Conv_IBW];
                stage1_weight[i] <= i_cnn_weight[i * `W_BW +: `W_BW];
            end
        end
    end

    //==============================================================================
    // DSP Multiplication Stage 2
    //==============================================================================
    reg signed [`M_BW-1:0] stage2_mul [0:KERNEL_SIZE-1];

    genvar mul_idx;
    generate
        for (mul_idx = 0; mul_idx < KERNEL_SIZE; mul_idx = mul_idx + 1) begin : gen_dsp_mul
            (* use_dsp = "yes" *)
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    stage2_mul[mul_idx] <= {`M_BW{1'b0}};
                end else if (r_valid[0]) begin
                    stage2_mul[mul_idx] <= stage1_fmap[mul_idx] * stage1_weight[mul_idx];
                end
            end
        end
    endgenerate

    // Debug: 2D array view
    reg signed [`M_BW-1:0] debug_mul_2d [0:`KY-1][0:`KX-1];
    integer j, k;
    always @(posedge clk) begin
        if (r_valid[1]) begin
            for(j = 0; j < `KY; j = j + 1) begin
                for(k = 0; k < `KX; k = k + 1) begin
                    debug_mul_2d[j][k] <= stage2_mul[j*`KX + k];
                end
            end
        end
    end

    //==============================================================================
    // Tree-based Partial Sum Stage 3 (5개의 5-input adders)
    //==============================================================================
    reg signed [`AK_BW-1:0] stage3_partial_sum [0:`KY-1];

    genvar row_idx;
    generate
        for (row_idx = 0; row_idx < `KY; row_idx = row_idx + 1) begin : gen_row_sum
            wire signed [`AK_BW-1:0] row_sum_temp;
            
            // 각 행의 5개 원소를 더함
            assign row_sum_temp = 
                {{(`AK_BW-`M_BW){stage2_mul[row_idx*`KX + 0][`M_BW-1]}}, stage2_mul[row_idx*`KX + 0]} +
                {{(`AK_BW-`M_BW){stage2_mul[row_idx*`KX + 1][`M_BW-1]}}, stage2_mul[row_idx*`KX + 1]} +
                {{(`AK_BW-`M_BW){stage2_mul[row_idx*`KX + 2][`M_BW-1]}}, stage2_mul[row_idx*`KX + 2]} +
                {{(`AK_BW-`M_BW){stage2_mul[row_idx*`KX + 3][`M_BW-1]}}, stage2_mul[row_idx*`KX + 3]} +
                {{(`AK_BW-`M_BW){stage2_mul[row_idx*`KX + 4][`M_BW-1]}}, stage2_mul[row_idx*`KX + 4]};

            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    stage3_partial_sum[row_idx] <= {`AK_BW{1'b0}};
                end else if (r_valid[1]) begin
                    stage3_partial_sum[row_idx] <= row_sum_temp;
                end
            end
        end
    endgenerate

    //==============================================================================
    // Two-level Final Accumulation Stage 4-5
    //==============================================================================
    // Stage 4: 5개를 3개+2개로 분할
    reg signed [`AK_BW-1:0] stage4_sum1, stage4_sum2, stage4_sum3;
    
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            stage4_sum1 <= {`AK_BW{1'b0}};
            stage4_sum2 <= {`AK_BW{1'b0}};
            stage4_sum3 <= {`AK_BW{1'b0}};
        end else if (r_valid[2]) begin
            stage4_sum1 <= stage3_partial_sum[0] + stage3_partial_sum[1];
            stage4_sum2 <= stage3_partial_sum[2] + stage3_partial_sum[3];
            stage4_sum3 <= stage3_partial_sum[4];
        end
    end

    // Stage 5: 최종 합산
    reg signed [`AK_BW-1:0] stage5_final_acc;
    
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            stage5_final_acc <= {`AK_BW{1'b0}};
        end else if (r_valid[3]) begin
            stage5_final_acc <= stage4_sum1 + stage4_sum2 + stage4_sum3;
        end
    end

    //==============================================================================
    // Output Assignment
    //==============================================================================
    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_kernel_acc = stage5_final_acc;

endmodule
