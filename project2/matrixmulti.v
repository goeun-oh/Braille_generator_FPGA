`include "timescale.vh"
`include "defines_cnn_core.vh"

module matrixmultiplex (
    input  wire                         clk,
    input  wire                         reset_n,
    input  wire                         i_in_valid,
    input  wire [`FC_IN_VEC*`OF_BW-1:0] i_in_fmap,
    output wire                         o_ot_valid,
    output wire [`CO*`OUT_BW-1:0]       o_ot_bias
);
    localparam LATENCY = 2; // 파이프라인 단계에 따라 조정

    // --- 1. Weight & Bias 선언 및 초기화 ---
    reg signed [(`CO * `FC_IN_VEC * `W_BW)-1:0] i_cnn_weight;
    reg signed [(`CO * `BIAS_BW)-1:0]           i_cnn_bias;

    initial begin
        // 3*48=144개 weight 초기화 (7'sd-11 형태로 공백 없이)
        i_cnn_weight = {
            7'sd5, -7'sd11, -7'sd15, 7'sd26,  7'sd31,  7'sd3,   -7'sd18, 7'sd1,
            7'sd28,  -7'sd16, -7'sd20, 7'sd5,   7'sd15,  -7'sd18,  -7'sd32, 7'sd3,
            7'sd1,   -7'sd10, 7'sd10,  7'sd6,   7'sd1,   7'sd20,   -7'sd19, -7'sd12,
            -7'sd18, 7'sd3,   -7'sd10, -7'sd27, 7'sd37,  7'sd11,   7'sd27,  -7'sd12,
            7'sd13,  -7'sd2,  -7'sd4,  7'sd5,   -7'sd14, 7'sd13,   -7'sd15, -7'sd6,
            7'sd2,   7'sd2,   -7'sd14, -7'sd17, 7'sd5,   -7'sd6,   7'sd5,   7'sd12,
            -7'sd20, 7'sd17,  7'sd26,  7'sd2,   -7'sd11, -7'sd10,  7'sd24,  7'sd23,
            7'sd0,   -7'sd25, 7'sd5,   -7'sd18, 7'sd9,   7'sd9,    7'sd15,  -7'sd17,
            7'sd11,  7'sd3,   7'sd15,  -7'sd2,  7'sd20,  -7'sd17,  -7'sd15, 7'sd0,
            7'sd27,  7'sd3,   -7'sd24, 7'sd7,   -7'sd21, -7'sd12,  -7'sd19, -7'sd22,
            7'sd8,   7'sd17,  -7'sd12, 7'sd8,   -7'sd15, 7'sd6,    7'sd3,   7'sd5,
            7'sd9,   7'sd5,   -7'sd11, 7'sd13,  -7'sd5,  -7'sd14,  7'sd5,   -7'sd14,
            7'sd20,  -7'sd3,  -7'sd14, -7'sd10, -7'sd20, -7'sd8,   -7'sd10, -7'sd18,
            -7'sd7,  7'sd21,  7'sd17,  7'sd13,  -7'sd24, -7'sd18,  -7'sd10, 7'sd5,
            7'sd0,   -7'sd3,  7'sd11,  -7'sd1,  -7'sd33, 7'sd13,   7'sd29,  7'sd22,
            -7'sd3,  7'sd1,   7'sd16,  7'sd23,  -7'sd5,  -7'sd18,  7'sd11,  -7'sd14,
            7'sd6,   -7'sd3,  -7'sd12, 7'sd7,   -7'sd8,  -7'sd2,   7'sd3,   -7'sd13,
            7'sd15,  7'sd2,   -7'sd6,  -7'sd14, -7'sd10, -7'sd17,  -7'sd5,  7'sd16
        };
        // 3개 bias 초기화 (6-bit)
        i_cnn_bias = {6'sd11, -6'sd7, -6'sd15};
    end

    // --- 2. Valid 신호 파이프라인 ---
    reg [LATENCY-1:0] r_valid;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) r_valid <= {LATENCY{1'b0}};
        else begin
            r_valid[LATENCY-2]  <= i_in_valid;
            r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
        end 
    end
    wire [LATENCY-1:0] ce = r_valid;

    // --- 3. Dot Product 및 Bias 덧셈 ---
    reg  [`CO*`OUT_BW-1:0] r_plus_bias;

    genvar i_co;
    generate
        for (i_co = 0; i_co < `CO; i_co = i_co + 1) begin : GEN_FC_NEURON
            // Step A: 병렬 곱셈
            wire signed [`FC_IN_VEC*`MUL_BW-1:0] products;
            genvar i_vec;
            for (i_vec = 0; i_vec < `FC_IN_VEC; i_vec = i_vec + 1) begin : GEN_MAC
                assign products[i_vec*`MUL_BW +: `MUL_BW] =
                    $signed(i_in_fmap[i_vec*`OF_BW +: `OF_BW]) *
                    $signed(i_cnn_weight[(i_co*`FC_IN_VEC + i_vec)*`W_BW +: `W_BW]);
            end

            // Step B: Adder Tree로 모든 곱셈 결과 합산
            reg signed [`ACC_BW-1:0] accumulated_sum;
            integer k;
            always @(*) begin
                accumulated_sum = 0;
                for (k = 0; k < `FC_IN_VEC; k = k + 1) begin
                    accumulated_sum = accumulated_sum + products[k*`MUL_BW +: `MUL_BW];
                end
            end

            // Step C: Bias 덧셈
            wire signed [`OUT_BW-1:0] final_output_neuron =
                accumulated_sum + $signed(i_cnn_bias[i_co*`BIAS_BW +: `BIAS_BW]);

            // Step D: 최종 결과 레지스터에 저장
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n)        r_plus_bias[i_co*`OUT_BW +: `OUT_BW] <= 0;
                else if (ce[0]) r_plus_bias[i_co*`OUT_BW +: `OUT_BW] <= final_output_neuron;
            end
        end
    endgenerate

    // --- 4. 최종 출력 할당 ---
    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_bias  = r_plus_bias;

endmodule