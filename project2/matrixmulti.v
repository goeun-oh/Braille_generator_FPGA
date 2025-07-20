`include "timescale.vh"
`include "defines_cnn_core.vh"

module matrixmultiplex (
    input  wire                       clk,
    input  wire                       reset_n,
    input  wire                       i_in_valid,
    input  wire [`FC_IN_VEC*`OF_BW-1:0]  i_in_fmap,    // 48 * 32-bit
    output wire                       o_ot_valid,
    output wire [`CO*`OUT_BW-1:0]        o_ot_bias     // 3 * 40-bit
);
    localparam LATENCY = 2; // 파이프라인 단계에 따라 조정

    // --- 1. Weight & Bias 선언 및 초기화 ---
    reg signed [(`CO * `FC_IN_VEC * `W_BW)-1:0] i_cnn_weight; // 3 * 48 * 7-bit
    reg signed [(`CO * `BIAS_BW)-1:0]           i_cnn_bias;   // 3 * 6-bit

    initial begin
        // 3*48=144개 weight 초기화 (-7'sd11 형태로 공백 없이)
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
        else          r_valid <= {r_valid[LATENCY-2], i_in_valid};
    end
    wire [LATENCY-1:0] ce = r_valid;

    // --- 3. Dot Product 및 Bias 덧셈 ---
    reg  [`CO*`OUT_BW-1:0] r_plus_bias;

    genvar i_co;
    generate
        for (i_co = 0; i_co < `CO; i_co = i_co + 1) begin : GEN_FC_NEURON
            // Step A: Parallel Multiplication
            wire signed [`FC_IN_VEC*`MUL_BW-1:0] products;
            genvar i_vec;
            for (i_vec = 0; i_vec < `FC_IN_VEC; i_vec = i_vec + 1) begin : GEN_MAC
                assign products[i_vec*`MUL_BW +: `MUL_BW] =
                    $signed(i_in_fmap[i_vec*`OF_BW +: `OF_BW]) *
                    $signed(i_cnn_weight[(i_co*`FC_IN_VEC + i_vec)*`W_BW +: `W_BW]);
            end

            // Step B: Adder Tree to sum all products (combinational)
            reg signed [`ACC_BW-1:0] accumulated_sum;
            integer k;
            always @(*) begin
                accumulated_sum = 0;
                for (k = 0; k < `FC_IN_VEC; k = k + 1) begin
                    accumulated_sum = accumulated_sum + products[k*`MUL_BW +: `MUL_BW];
                end
            end

            // Step C: Add bias (combinational)
            wire signed [`OUT_BW-1:0] final_output_neuron =
                accumulated_sum + $signed(i_cnn_bias[i_co*`BIAS_BW +: `BIAS_BW]);

            // Step D: Register the final result
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n)        r_plus_bias[i_co*`OUT_BW +: `OUT_BW] <= 0;
                else if (i_in_valid) r_plus_bias[i_co*`OUT_BW +: `OUT_BW] <= final_output_neuron;
            end
        end
    endgenerate

    // --- 4. 최종 출력 할당 ---
    assign o_ot_valid = r_valid[LATENCY-1]; // Adjust latency based on MAC pipeline stages
    assign o_ot_bias  = r_plus_bias;

endmodule

// `include "defines_cnn_core.vh"

// module matrixmultiplex (
//     // clk,
//     // reset_n,
//     // i_in_valid,
//     // // i_cnn_weight,
//     // i_in_fmap,
//     // o_ot_valid,
//     // o_ot_bias 
//     input                       clk,
//     input                       reset_n,
//     input                       i_in_valid,
//     // input  [`CI*3*48*7-1:0]     i_cnn_weight;
//     input  [`CI*4*4*32-1:0]     i_in_fmap,
//     output                      o_ot_valid,
//     output [(`CI*40)-1:0]       o_ot_bias
// );
    
//     localparam LATENCY = 2;

//     reg signed [(3*48*7)-1:0]     i_cnn_weight;
//         // 초기화는 initial 블록에서 수행

//     reg signed [(3*7)-1 : 0]      i_cnn_bias;

//     initial begin
//         i_cnn_weight = {7'sd5, -7'sd11,  -7'sd15, 7'sd26,  7'sd31,  7'sd3,   -7'sd18, 
//             7'sd1, 7'sd28,  -7'sd16,  -7'sd20, 7'sd5,   7'sd15,  -7'sd18,  -7'sd32, 7'sd3,
//             7'sd1,   -7'sd10,  7'sd10,  7'sd6,   7'sd1,   7'sd20,  -7'sd19, -7'sd12,
//             -7'sd18, 7'sd3,    -7'sd10, -7'sd27, 7'sd37,  7'sd11,  7'sd27,  -7'sd12,
//             7'sd13,  -7'sd2,   -7'sd4,  7'sd5,   -7'sd14, 7'sd13,  -7'sd15, -7'sd6,
//             7'sd2,   7'sd2,    -7'sd14, -7'sd17, 7'sd5,   -7'sd6,   7'sd5,   7'sd12,
//             -7'sd20, 7'sd17,   7'sd26,  7'sd2,   -7'sd11, -7'sd10,  7'sd24,  7'sd23,
//             7'sd0,   -7'sd25,  7'sd5,   -7'sd18, 7'sd9,   7'sd9,    7'sd15,  -7'sd17,
//             7'sd11,  7'sd3,    7'sd15,  -7'sd2,  7'sd20,  -7'sd17,  -7'sd15, 7'sd0,
//             7'sd27,  7'sd3,    -7'sd24, 7'sd7,   -7'sd21, -7'sd12,  -7'sd19, -7'sd22,
//             7'sd8,   7'sd17,   -7'sd12, 7'sd8,   -7'sd15, 7'sd6,    7'sd3,   7'sd5,
//             7'sd9,   7'sd5,    -7'sd11, 7'sd13,  -7'sd5,  -7'sd14,  7'sd5,   -7'sd14,
//             7'sd20,  -7'sd3,   -7'sd14, -7'sd10, -7'sd20, -7'sd8,   -7'sd10, -7'sd18,
//             -7'sd7,  7'sd21,   7'sd17,  7'sd13,  -7'sd24, -7'sd18,  -7'sd10, 7'sd5,
//             7'sd0,   -7'sd3,   7'sd11,  -7'sd1,  -7'sd33, 7'sd13,   7'sd29,  7'sd22,
//             -7'sd3,  7'sd1,    7'sd16,  7'sd23,  -7'sd5,  -7'sd18,  7'sd11,  -7'sd14,
//             7'sd6,   -7'sd3,   -7'sd12, 7'sd7,   -7'sd8,  -7'sd2,   7'sd3,   -7'sd13,
//             7'sd15,  7'sd2,    -7'sd6,  -7'sd14, -7'sd10, -7'sd17,  -7'sd5,   7'sd16
//         };
//         i_cnn_bias = {7'sd11, -7'sd7 ,-7'sd15};        
//     end



// //==============================================================================
// // Data Enable Signals 
// //==============================================================================
//     wire    [LATENCY-1 : 0] 	ce;
//     reg     [LATENCY-1 : 0] 	r_valid;
//     always @(posedge clk or negedge reset_n) begin
//         if(!reset_n) begin
//             r_valid   <= {LATENCY{1'b0}};
//         end else begin
//             r_valid[LATENCY-2]  <= i_in_valid;
//             r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
//         end
//     end

//     assign	ce = r_valid;


// //==============================================================================
// // mul = fmap * weight
// //==============================================================================

// wire      [`CO*39-1 : 0]    mul  ;
// reg       [`CO*39-1 : 0]    r_mul;

// genvar mul_idx;
// generate
// 	for(mul_idx = 0; mul_idx < `CO; mul_idx = mul_idx + 1) begin : gen_mul
//     // 48개 * 7bit
// 		assign  mul[mul_idx * 39 +: 39]	= i_in_fmap * i_cnn_weight[mul_idx * 48 * 7 +: 48 * 7];
	
// 		always @(posedge clk or negedge reset_n) begin
// 		    if(!reset_n) begin
// 		        r_mul[mul_idx * 39 +: 39] <= {39{1'b0}};
// 		    end else if(i_in_valid)begin
// 		        r_mul[mul_idx * 39 +: 39] <= mul[mul_idx * 39 +: 39];
// 		    end
// 		end
// 	end
// endgenerate

// // bias 덧셈 +1bit 한 이유 덧셈 시 큰 bit + 1만 됨 39+6이 아님을 확인
// reg       [`CO*40-1 : 0]    w_plus_bias;
// reg       [`CO*40-1 : 0]    r_plus_bias;

// integer bias_idx;
// generate
// 	always @ (*) begin
// 		w_plus_bias[0 +: `CO*40]= {`CO*40{1'b0}};
// 		for(bias_idx =0; bias_idx < `CO; bias_idx = bias_idx +1) begin
// 			w_plus_bias[`CO*40 +: 40] = r_mul[bias_idx*39 +: 39] + $signed(i_cnn_bias[bias_idx*7 +: 7]); 
// 		end
// 	end
// 	always @(posedge clk or negedge reset_n) begin
// 	    if(!reset_n) begin
// 	        r_plus_bias[0 +: `CO*40] <= {`CO*40{1'b0}};
// 	    end else if(ce[LATENCY-2])begin
// 	        r_plus_bias[0 +: `CO*40] <= w_plus_bias[0 +: `CO*40];
// 	    end
// 	end
// endgenerate

// assign o_ot_valid = r_valid[LATENCY-1];
// assign o_ot_bias = r_plus_bias;

// endmodule
