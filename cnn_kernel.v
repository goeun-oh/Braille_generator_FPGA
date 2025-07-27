`timescale 1ns / 1ps


module cnn_kernel #(
    parameter KX = 5,  // Number of Kernel X
    parameter KY = 5,  // Number of Kernel Y
    parameter I_F_BW = 8,  // Bit Width of Input Feature
    parameter W_BW = 8,  // BW of weight parameter
    parameter B_BW = 16,  // BW of bias parameter
    parameter AK_BW = 21,  // M_BW + log(KY*KX) Accum Kernel 
    parameter M_BW = 16 // I_F_BW * W_BW
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
            r_valid[LATENCY-2] <= i_in_valid;
            r_valid[LATENCY-1] <= r_valid[LATENCY-2];
        end
    end

    assign ce = r_valid;

    //==============================================================================
    // mul = fmap * weight
    //==============================================================================

    wire [KY*KX*M_BW-1 : 0] mul;
    reg [KY*KX*M_BW-1 : 0] r_mul;

    // TODO Multiply each of Kernels
    genvar mul_idx;
    generate
        for (
            mul_idx = 0; mul_idx < KY * KX; mul_idx = mul_idx + 1
        ) begin : gen_mul
            assign  mul[mul_idx * M_BW +: M_BW]	=$signed({1'b0, i_in_fmap[mul_idx * I_F_BW +: I_F_BW]}) * $signed(i_cnn_weight[mul_idx * W_BW +: W_BW]);
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_mul[mul_idx*M_BW+:M_BW] <= 0;
                end else if (i_in_valid) begin
                    r_mul[mul_idx*M_BW+:M_BW] <= $signed(mul[mul_idx*M_BW+:M_BW]);
                end
            end
        end
    endgenerate
    reg signed [M_BW-1:0] reg_r_mul [0:KY-1][0:KX-1];
    always @(posedge clk) begin
        if(i_in_valid)begin
            for (k= 0; k < KY; k = k + 1) begin
                for (j= 0; j < KX; j = j + 1) begin
                    reg_r_mul[k][j] <= $signed(r_mul[(k*KY+j)*M_BW +: M_BW]);
                end
            end
        end
    end
    reg signed [AK_BW-1 : 0] acc_kernel;
    reg signed [AK_BW-1 : 0] r_acc_kernel;

    integer               acc_idx;
    generate
        always @(*) begin
            acc_kernel[0+:AK_BW] = 0;
            for (acc_idx = 0; acc_idx < KY * KX; acc_idx = acc_idx + 1) begin
                acc_kernel[0 +: AK_BW] = $signed(acc_kernel[0 +: AK_BW]) + $signed(r_mul[acc_idx*M_BW +: M_BW]);
            end
        end
    endgenerate
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_acc_kernel[0+:AK_BW] <= 0;
        end else if (ce[LATENCY-2]) begin
            r_acc_kernel[0+:AK_BW] <= $signed(acc_kernel[0+:AK_BW]);
        end
    end
    integer j;
    integer k;
    reg [W_BW-1:0] reg_weight [0:KY-1][0:KX-1];
    always @(posedge clk) begin
        for (k= 0; k < KY; k = k + 1) begin
            for (j= 0; j < KX; j = j + 1) begin
                reg_weight[k][j] <= i_cnn_weight[(k*KY+j)*W_BW +: W_BW];
            end
        end
    end
    reg [I_F_BW-1:0] reg_i_fmap [0:KY-1][0:KX-1];
    always @(posedge clk) begin
        if(i_in_valid)begin
            for (k= 0; k < KY; k = k + 1) begin
                for (j= 0; j < KX; j = j + 1) begin
                    reg_i_fmap[k][j] <= i_in_fmap[(k*KY+j)*I_F_BW +: I_F_BW];
                end
            end
        end
    end
    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_kernel_acc = r_acc_kernel;

    
endmodule
