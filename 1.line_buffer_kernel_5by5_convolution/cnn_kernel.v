`timescale 1ns / 1ps


module cnn_kernel #(
    parameter KX = 5,  // Number of Kernel X
    parameter KY = 5,  // Number of Kernel Y
    parameter I_F_BW = 8,  // Bit Width of Input Feature
    parameter W_BW = 7,  // BW of weight parameter
    parameter B_BW = 7,  // BW of bias parameter
    parameter AK_BW = 20,  // M_BW + log(KY*KX) Accum Kernel 
    parameter M_BW = 15 // I_F_BW * W_BW
)(
    // Clock & Reset
    input clk,
    input reset_n,
    input [KX*KY*W_BW-1 : 0] i_cnn_weight,
    input i_in_valid,
    input [KX*KY*I_F_BW-1 : 0] i_in_fmap,
    output o_ot_valid,
    output [AK_BW-1 : 0] o_ot_kernel_acc
);


    localparam LATENCY = 1;


    //==============================================================================
    // Data Enable Signals 
    //==============================================================================
    //shift register
    //입력이 들어오면 r_valid가 1.
    //곱셈 연산에 한 클럭 소모
    //누산(accumulation) 연산에 1클럭 소모
    //총 2클럭의 지연이 있어서 valid 신호도 그만큼 지연되어야함
    wire [LATENCY-1 : 0] ce;
    reg  [LATENCY-1 : 0] r_valid;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[LATENCY-1] <= i_in_valid;
            //r_valid[LATENCY-1] <= r_valid[LATENCY-2];
        end
    end

    assign ce = r_valid;

    //==============================================================================
    // mul = fmap * weight
    //==============================================================================

    wire [KY*KX*M_BW-1 : 0] mul;
    reg  [KY*KX*M_BW-1 : 0] r_mul;

    // TODO Multiply each of Kernels
    genvar mul_idx;
    generate
        for (
            mul_idx = 0; mul_idx < KY * KX; mul_idx = mul_idx + 1
        ) begin : gen_mul
            assign  mul[mul_idx * M_BW +: M_BW]	= i_in_fmap[mul_idx * I_F_BW +: I_F_BW] * i_cnn_weight[mul_idx * W_BW +: W_BW];

            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_mul[mul_idx*M_BW+:M_BW] <= {M_BW{1'b0}};
                end else if (i_in_valid) begin
                    r_mul[mul_idx*M_BW+:M_BW] <= mul[mul_idx*M_BW+:M_BW];
                end
            end
        end
    endgenerate
    reg     [AK_BW-1 : 0] acc_kernel;
    reg     [AK_BW-1 : 0] r_acc_kernel;

    integer               acc_idx;
    generate
        always @(*) begin
            acc_kernel[0+:AK_BW] = {AK_BW{1'b0}};
            for (acc_idx = 0; acc_idx < KY * KX; acc_idx = acc_idx + 1) begin
                acc_kernel[0 +: AK_BW] = acc_kernel[0 +: AK_BW] + r_mul[acc_idx*M_BW +: M_BW];
            end
        end
        always @(posedge clk or negedge reset_n) begin
            if (!reset_n) begin
                r_acc_kernel[0+:AK_BW] <= {AK_BW{1'b0}};
            end else if (ce[LATENCY-1]) begin
                r_acc_kernel[0+:AK_BW] <= acc_kernel[0+:AK_BW];
            end
        end
    endgenerate

    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_kernel_acc = r_acc_kernel;

endmodule
