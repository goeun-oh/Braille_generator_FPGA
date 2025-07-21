`include "timescale.vh"
`include "defines_cnn_core.vh"

// module matrixmultiplex (
//     input  wire                          clk,
//     input  wire                          reset_n,
//     input  wire                          i_in_valid,
//     input  wire [`FC_IN_VEC*`OF_BW-1:0]  i_in_fmap,
//     output wire                          o_ot_valid,
//     output reg  [`CO*`OUT_BW-1:0]        o_ot_bias // reg 타입으로 변경
// );
//     localparam LATENCY = 2;

//     // --- 1. Weight & Bias 선언 및 초기화 ---
//     reg signed [(`CO * `FC_IN_VEC * `W_BW)-1:0] i_cnn_weight;
//     reg signed [(`CO * `BIAS_BW)-1:0]           i_cnn_bias;
//     initial begin
//         // (가중치, 편향 초기화는 그대로 유지)
//         i_cnn_weight = {
//             7'sd5, -7'sd11, -7'sd15, 7'sd26,  7'sd31,  7'sd3,   -7'sd18, 7'sd1,
//             7'sd28,  -7'sd16, -7'sd20, 7'sd5,   7'sd15,  -7'sd18,  -7'sd32, 7'sd3,
//             7'sd1,   -7'sd10, 7'sd10,  7'sd6,   7'sd1,   7'sd20,   -7'sd19, -7'sd12,
//             -7'sd18, 7'sd3,   -7'sd10, -7'sd27, 7'sd37,  7'sd11,   7'sd27,  -7'sd12,
//             7'sd13,  -7'sd2,  -7'sd4,  7'sd5,   -7'sd14, 7'sd13,   -7'sd15, -7'sd6,
//             7'sd2,   7'sd2,   -7'sd14, -7'sd17, 7'sd5,   -7'sd6,   7'sd5,   7'sd12,
//             -7'sd20, 7'sd17,  7'sd26,  7'sd2,   -7'sd11, -7'sd10,  7'sd24,  7'sd23,
//             7'sd0,   -7'sd25, 7'sd5,   -7'sd18, 7'sd9,   7'sd9,    7'sd15,  -7'sd17,
//             7'sd11,  7'sd3,   7'sd15,  -7'sd2,  7'sd20,  -7'sd17,  -7'sd15, 7'sd0,
//             7'sd27,  7'sd3,   -7'sd24, 7'sd7,   -7'sd21, -7'sd12,  -7'sd19, -7'sd22,
//             7'sd8,   7'sd17,  -7'sd12, 7'sd8,   -7'sd15, 7'sd6,    7'sd3,   7'sd5,
//             7'sd9,   7'sd5,   -7'sd11, 7'sd13,  -7'sd5,  -7'sd14,  7'sd5,   -7'sd14,
//             7'sd20,  -7'sd3,  -7'sd14, -7'sd10, -7'sd20, -7'sd8,   -7'sd10, -7'sd18,
//             -7'sd7,  7'sd21,  7'sd17,  7'sd13,  -7'sd24, -7'sd18,  -7'sd10, 7'sd5,
//             7'sd0,   -7'sd3,  7'sd11,  -7'sd1,  -7'sd33, 7'sd13,   7'sd29,  7'sd22,
//             -7'sd3,  7'sd1,   7'sd16,  7'sd23,  -7'sd5,  -7'sd18,  7'sd11,  -7'sd14,
//             7'sd6,   -7'sd3,  -7'sd12, 7'sd7,   -7'sd8,  -7'sd2,   7'sd3,   -7'sd13,
//             7'sd15,  7'sd2,   -7'sd6,  -7'sd14, -7'sd10, -7'sd17,  -7'sd5,  7'sd16
//         };
//         // 3개 bias 초기화 (6-bit)
//         i_cnn_bias = {6'sd11, -6'sd7, -6'sd15};

//     end

//     // --- 2. Valid 신호 파이프라인 (안정적인 코드로 수정) ---
//     reg [LATENCY-1:0] r_valid;
//     integer i;
//     always @(posedge clk or negedge reset_n) begin
//         if (!reset_n) begin
//             r_valid <= {LATENCY{1'b0}};
//         end else begin
//             r_valid[0] <= i_in_valid;
//             for (i = 1; i < LATENCY; i = i + 1) begin
//                 r_valid[i] <= r_valid[i-1];
//             end
//         end 
//     end

//     // --- 3. Dot Product 및 Bias 덧셈 ---
//     genvar i_co;
//     integer k;
//     generate
//         for (i_co = 0; i_co < `CO; i_co = i_co + 1) begin : GEN_FC_NEURON
            
//             // --- 파이프라인 레지스터 ---
//             reg signed [`ACC_BW-1:0] r_dot_product; // 1단계 파이프라인 레지스터

//             // --- Combinational Logic (조합 논리) ---
//             wire signed [`FC_IN_VEC*`MUL_BW-1:0] products;
//             wire signed [`ACC_BW-1:0] w_dot_sum;

//             // Step A: 병렬 곱셈
//             genvar i_vec;
//             for (i_vec = 0; i_vec < `FC_IN_VEC; i_vec = i_vec + 1) begin : GEN_MAC
//                 assign products[i_vec*`MUL_BW +: `MUL_BW] =
//                     $signed(i_in_fmap[i_vec*`OF_BW +: `OF_BW]) *
//                     $signed(i_cnn_weight[(i_co*`FC_IN_VEC + i_vec)*`W_BW +: `W_BW]);
//             end

//             // Step B: 곱셈 결과 합산 (조합 논리로 구현)
//             // 시뮬레이션과 합성을 모두 고려한 안정적인 합산 로직
//             integer k;
//             always @(*) begin
//                 reg signed [`ACC_BW-1:0] temp_sum;
//                 temp_sum = 0;
//                 for (k = 0; k < `FC_IN_VEC; k = k + 1) begin
//                     temp_sum = temp_sum + products[k*`MUL_BW +: `MUL_BW];
//                 end
//                 // always 블록의 최종 결과는 wire에 할당할 수 없으므로,
//                 // 이 블록 밖에서 할당하거나, SystemVerilog의 always_comb를 사용해야 합니다.
//                 // 가장 확실한 방법은 아래와 같이 파이프라인 1단계에서 직접 계산하는 것입니다.
//             end
            
//             // --- Sequential Logic (순차 논리: 파이프라인 구현) ---
//             integer j;
//             always @(posedge clk or negedge reset_n) begin
//                 if (!reset_n) begin
//                     r_dot_product <= 0;
//                     // 출력 레지스터 o_ot_bias도 리셋
//                     o_ot_bias[i_co*`OUT_BW +: `OUT_BW] <= 0;
//                 end else begin
//                     // 파이프라인 1단계: 입력이 유효할 때, 곱셈의 합을 계산하여 저장
//                     if (i_in_valid) begin
//                         // always 블록 내에서 임시 변수 사용
//                         reg signed [`ACC_BW-1:0] temp_sum_seq;
//                         temp_sum_seq = 0;
//                         for (j = 0; j < `FC_IN_VEC; j = j + 1) begin
//                             temp_sum_seq = temp_sum_seq + products[j*`MUL_BW +: `MUL_BW];
//                         end
//                         r_dot_product <= temp_sum_seq;
//                     end
                    
//                     // 파이프라인 2단계: 1단계의 valid 신호(r_valid[0])가 유효할 때, 편향을 더해 최종 출력에 저장
//                     if (r_valid[0]) begin
//                         o_ot_bias[i_co*`OUT_BW +: `OUT_BW] <= r_dot_product + $signed(i_cnn_bias[i_co*`BIAS_BW +: `BIAS_BW]);
//                     end
//                 end
//             end
//         end
//     endgenerate

//     // --- 4. 최종 출력 Valid 신호 할당 ---
//     assign o_ot_valid = r_valid[LATENCY-1];

// endmodule
module matrixmultiplex (
    input  wire                          clk,
    input  wire                          reset_n,
    input  wire                          i_in_valid,
    input  wire [`FC_IN_VEC*`OF_BW-1:0]  i_in_fmap,
    output wire                          o_ot_valid,
    output reg  [`CO*`OUT_BW-1:0]        o_ot_bias // reg 타입으로 변경
);
    localparam LATENCY = 3;

    // --- 1. Weight & Bias 선언 및 초기화 ---
    reg signed [(`CO * `FC_IN_VEC * `W_BW)-1:0] i_cnn_weight;
    reg signed [(`CO * `BIAS_BW)-1:0]           i_cnn_bias;
    initial begin
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
        // MSB = 7'sd5 LSB = 7'sd16
        // 3개 bias 초기화 (6-bit)
        i_cnn_bias = {6'sd11, -6'sd7, -6'sd15};
        // N2=11, N1=-7, N0=-15
    end

    reg [LATENCY-1:0] r_valid;
    integer i;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[0] <= i_in_valid;
            for (i = 1; i < LATENCY; i = i + 1) begin
                r_valid[i] <= r_valid[i-1];
            end
        end 
    end

    // kernal 1px 7bit 48 * 3과 iinfmap 1px 32bit 48 * 1 과 곱하는 함수
    // 결과 값 1px 39bit 48*3 이 생김 
    
    wire      [`CI*`FC_IN_VEC*`MUL_BW-1 : 0]    mul  ;
    reg       [`CI*`FC_IN_VEC*`MUL_BW-1 : 0]    r_mul;

    genvar ci, mul_idx;
    generate
    for(ci = 0; ci < `CI; ci = ci + 1) begin : layer
        for(mul_idx = 0; mul_idx < `FC_IN_VEC; mul_idx = mul_idx + 1) begin : gen_mul
            assign  mul[mul_idx * `MUL_BW +: `MUL_BW]	= i_in_fmap[mul_idx * `IF_BW +: `IF_BW] * i_cnn_weight[mul_idx * `W_BW +: `W_BW];
        
            always @(posedge clk or negedge reset_n) begin
                if(!reset_n) begin
                    r_mul[mul_idx * `MUL_BW +: `MUL_BW] <= {`MUL_BW{1'b0}};
                end else if(i_in_valid)begin
                    r_mul[mul_idx * `MUL_BW +: `MUL_BW] <= mul[mul_idx * `MUL_BW +: `MUL_BW];
                end
            end
        end
    end
    endgenerate

    reg       [`CO*`ACC_BW-1 : 0]    acc 	;
    reg       [`CO*`ACC_BW-1 : 0]    r_acc   ;
    
    genvar co; 
    integer acc_idx;
    generate
        for(co = 0; co < `CO; co = co + 1) begin
            always @ (*) begin
                acc[co*`ACC_BW +: `ACC_BW] = {`ACC_BW{1'b0}};
                for(acc_idx =0; acc_idx < `FC_IN_VEC; acc_idx = acc_idx +1) begin
                    acc[co*`ACC_BW +: `ACC_BW] = acc[co*`ACC_BW +: `ACC_BW] + r_mul[acc_idx*`MUL_BW +: `MUL_BW]; 
                end
            end
            always @(posedge clk or negedge reset_n) begin
                if(!reset_n) begin
                    r_acc[co*`ACC_BW +: `ACC_BW] <= {`ACC_BW{1'b0}};
                    // 아래 ce 파트 수정 필요
                end else if(r_valid[0])begin
                    r_acc[co*`ACC_BW +: `ACC_BW] <= acc[co*`ACC_BW +: `ACC_BW];
                end
            end
        end
    endgenerate

    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_kernel_acc = r_acc;  


    endmodule