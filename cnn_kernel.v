// 기존 dsp_multiplier 모듈을 LUT 버전으로 대체
module lut_multiplier #(
    parameter A_BW = 9,
    parameter B_BW = 8,
    parameter P_BW = 16
)(
    input wire clk,
    input wire reset_n,
    input wire enable,
    input wire signed [A_BW-1:0] a,
    input wire signed [B_BW-1:0] b,
    output reg signed [P_BW-1:0] product
);

    // DSP 사용하지 않는 순수 LUT 곱셈
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            product <= {P_BW{1'b0}};
        end else if (enable) begin
            product <= a * b;  // 자동으로 LUT 사용
        end
    end

endmodule

// cnn_kernel.v 수정 버전
module cnn_kernel #(
    parameter KX = 5,
    parameter KY = 5,
    parameter I_F_BW = 8,
    parameter W_BW = 8,
    parameter B_BW = 16,
    parameter AK_BW = 21,
    parameter M_BW = 16
)(
    input clk,
    input reset_n,
    input [KX*KY*W_BW-1 : 0] i_cnn_weight,
    input i_in_valid,
    input [KX*KY*I_F_BW-1 : 0] i_in_fmap,
    output o_ot_valid,
    output signed [AK_BW-1 : 0] o_ot_kernel_acc
);

    localparam LATENCY = 4;  // LUT 곱셈으로 인한 지연 증가
    localparam KERNEL_SIZE = KX * KY;

    reg [LATENCY-1 : 0] r_valid;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid <= {r_valid[LATENCY-2:0], i_in_valid};
        end
    end

    // Input pipelining
    reg [I_F_BW-1:0] r_fmap [0:KERNEL_SIZE-1];
    reg [W_BW-1:0] r_weight [0:KERNEL_SIZE-1];

    integer i;
    always @(posedge clk) begin
        if (i_in_valid) begin
            for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                r_fmap[i] <= i_in_fmap[i * I_F_BW +: I_F_BW];
                r_weight[i] <= i_cnn_weight[i * W_BW +: W_BW];
            end
        end
    end

    // LUT 곱셈기 인스턴스 (DSP 대신 LUT 사용)
    wire signed [M_BW-1:0] mul_results [0:KERNEL_SIZE-1];

    genvar mul_idx;
    generate
        for (mul_idx = 0; mul_idx < KERNEL_SIZE; mul_idx = mul_idx + 1) begin : gen_lut_mul
            lut_multiplier #(
                .A_BW(I_F_BW + 1),
                .B_BW(W_BW),
                .P_BW(M_BW)
            ) U_lut_mul (
                .clk(clk),
                .reset_n(reset_n),
                .enable(r_valid[0]),
                .a({1'b0, r_fmap[mul_idx]}),
                .b($signed(r_weight[mul_idx])),
                .product(mul_results[mul_idx])
            );
        end
    endgenerate

    // 누산도 LUT 사용 (DSP 사용 안함)
    reg signed [AK_BW-1:0] acc_temp;
    reg signed [AK_BW-1:0] r_acc_kernel;

    always @(*) begin
        acc_temp = {AK_BW{1'b0}};
        for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
            acc_temp = acc_temp + {{(AK_BW-M_BW){mul_results[i][M_BW-1]}}, mul_results[i]};
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_acc_kernel <= {AK_BW{1'b0}};
        end else if (r_valid[2]) begin
            r_acc_kernel <= acc_temp;
        end
    end

    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_kernel_acc = r_acc_kernel;

endmodule