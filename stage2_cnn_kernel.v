
`timescale 1ns / 1ps
`include "stage2_defines_cnn_core.v"
module stage2_cnn_kernel (
    input                               		   clk,
    input                               		   reset_n,
    input     signed [`KX*`KY*`W_BW-1 : 0] 	       i_cnn_weight,
    input                                          i_in_valid,
    input     signed [`KX*`KY*`ST2_Conv_IBW-1 : 0] i_in_fmap,
    output                                         o_ot_valid,
    output    signed [`AK_BW-1 : 0]  			   o_ot_kernel_acc           
);

    localparam LATENCY = 5;
    localparam KERNEL_SIZE = `KX * `KY;  // 25

    reg [LATENCY-1:0] r_valid;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid <= {r_valid[LATENCY-2:0], i_in_valid};
        end
    end

    // Input pipeline
    reg signed [`ST2_Conv_IBW-1:0] stage1_fmap [0:KERNEL_SIZE-1];
    reg signed [`W_BW-1:0] stage1_weight [0:KERNEL_SIZE-1];

    integer i;
    always @(posedge clk) begin
        if (i_in_valid) begin
            for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                stage1_fmap[i] <= i_in_fmap[i * `ST2_Conv_IBW +: `ST2_Conv_IBW];
                stage1_weight[i] <= i_cnn_weight[i * `W_BW +: `W_BW];
            end
        end
    end

    // 선택적 DSP 사용: 처음 10개만 DSP, 나머지 15개는 LUT
    reg signed [`M_BW-1:0] stage2_mul [0:KERNEL_SIZE-1];

    genvar mul_idx;
    generate
        for (mul_idx = 0; mul_idx < KERNEL_SIZE; mul_idx = mul_idx + 1) begin : gen_mixed_mul
            if (mul_idx < 10) begin : gen_dsp_mul
                // 처음 10개는 DSP 사용
                (* use_dsp = "yes" *)
                always @(posedge clk or negedge reset_n) begin
                    if (!reset_n) begin
                        stage2_mul[mul_idx] <= {`M_BW{1'b0}};
                    end else if (r_valid[0]) begin
                        stage2_mul[mul_idx] <= stage1_fmap[mul_idx] * stage1_weight[mul_idx];
                    end
                end
            end else begin : gen_lut_mul
                // 나머지 15개는 LUT 사용 (DSP 지시어 없음)
                always @(posedge clk or negedge reset_n) begin
                    if (!reset_n) begin
                        stage2_mul[mul_idx] <= {`M_BW{1'b0}};
                    end else if (r_valid[0]) begin
                        stage2_mul[mul_idx] <= stage1_fmap[mul_idx] * stage1_weight[mul_idx];
                    end
                end
            end
        end
    endgenerate

    // 나머지 누산 로직은 동일 (기존 코드 유지)
    reg signed [`AK_BW-1:0] stage3_partial_sum [0:`KY-1];

    genvar row_idx;
    generate
        for (row_idx = 0; row_idx < `KY; row_idx = row_idx + 1) begin : gen_row_sum
            wire signed [`AK_BW-1:0] row_sum_temp = 
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

    reg signed [`AK_BW-1:0] stage5_final_acc;
    
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            stage5_final_acc <= {`AK_BW{1'b0}};
        end else if (r_valid[3]) begin
            stage5_final_acc <= stage4_sum1 + stage4_sum2 + stage4_sum3;
        end
    end

    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_kernel_acc = stage5_final_acc;

endmodule