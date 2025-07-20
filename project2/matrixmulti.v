
`include "defines_cnn_core.vh"

module matrixmultiplex (
    // clk,
    // reset_n,
    // i_in_valid,
    // // i_cnn_weight,
    // i_in_fmap,
    // o_ot_valid,
    // o_ot_bias 
    input                       clk,
    input                       reset_n,
    input                       i_in_valid,
    // input  [`CI*3*48*7-1:0]     i_cnn_weight;
    input  [`CI*4*4*32-1:0]     i_in_fmap,
    output                      o_ot_valid,
    output [(`CI*40)-1:0]       o_ot_bias
);
    
    localparam LATENCY = 2;

    reg signed [(3*48*7)-1:0]     i_cnn_weight;
        // 초기화는 initial 블록에서 수행

    reg signed [(3*7)-1 : 0]      i_cnn_bias;

    initial begin
        i_cnn_weight = {7'sd5, 7'sd-11,  7'sd-15, 7'sd26,  7'sd31,  7'sd3,   7'sd-18, 
            7'sd1, 7'sd28,  7'sd-16,  7'sd-20, 7'sd5,   7'sd15,  7'sd-18,  7'sd-32, 7'sd3,
            7'sd1,   7'sd-10,  7'sd10,  7'sd6,   7'sd1,   7'sd20,  7'sd-19, 7'sd-12,
            7'sd-18, 7'sd3,    7'sd-10, 7'sd-27, 7'sd37,  7'sd11,  7'sd27,  7'sd-12,
            7'sd13,  7'sd-2,   7'sd-4,  7'sd5,   7'sd-14, 7'sd13,  7'sd-15, 7'sd-6,
            7'sd2,   7'sd2,    7'sd-14, 7'sd-17, 7'sd5,   7'sd-6,   7'sd5,   7'sd12,
            7'sd-20, 7'sd17,   7'sd26,  7'sd2,   7'sd-11, 7'sd-10,  7'sd24,  7'sd23,
            7'sd0,   7'sd-25,  7'sd5,   7'sd-18, 7'sd9,   7'sd9,    7'sd15,  7'sd-17,
            7'sd11,  7'sd3,    7'sd15,  7'sd-2,  7'sd20,  7'sd-17,  7'sd-15, 7'sd0,
            7'sd27,  7'sd3,    7'sd-24, 7'sd7,   7'sd-21, 7'sd-12,  7'sd-19, 7'sd-22,
            7'sd8,   7'sd17,   7'sd-12, 7'sd8,   7'sd-15, 7'sd6,    7'sd3,   7'sd5,
            7'sd9,   7'sd5,    7'sd-11, 7'sd13,  7'sd-5,  7'sd-14,  7'sd5,   7'sd-14,
            7'sd20,  7'sd-3,   7'sd-14, 7'sd-10, 7'sd-20, 7'sd-8,   7'sd-10, 7'sd-18,
            7'sd-7,  7'sd21,   7'sd17,  7'sd13,  7'sd-24, 7'sd-18,  7'sd-10, 7'sd5,
            7'sd0,   7'sd-3,   7'sd11,  7'sd-1,  7'sd-33, 7'sd13,   7'sd29,  7'sd22,
            7'sd-3,  7'sd1,    7'sd16,  7'sd23,  7'sd-5,  7'sd-18,  7'sd11,  7'sd-14,
            7'sd6,   7'sd-3,   7'sd-12, 7'sd7,   7'sd-8,  7'sd-2,   7'sd3,   7'sd-13,
            7'sd15,  7'sd2,    7'sd-6,  7'sd-14, 7'sd-10, 7'sd-17,  7'sd-5,   7'sd16
        };
        i_cnn_bias = {7'sd11, 7'sd-7 ,7'sd-15};        
    end



//==============================================================================
// Data Enable Signals 
//==============================================================================
    wire    [LATENCY-1 : 0] 	ce;
    reg     [LATENCY-1 : 0] 	r_valid;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid   <= {LATENCY{1'b0}};
        end else begin
            r_valid[LATENCY-2]  <= i_in_valid;
            r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
        end
    end

    assign	ce = r_valid;


//==============================================================================
// mul = fmap * weight
//==============================================================================

wire      [`CO*39-1 : 0]    mul  ;
reg       [`CO*39-1 : 0]    r_mul;

genvar mul_idx;
generate
	for(mul_idx = 0; mul_idx < `CO; mul_idx = mul_idx + 1) begin : gen_mul
    // 48개 * 7bit
		assign  mul[mul_idx * 39 +: 39]	= i_in_fmap * i_cnn_weight[mul_idx * 48 * 7 +: 48 * 7];
	
		always @(posedge clk or negedge reset_n) begin
		    if(!reset_n) begin
		        r_mul[mul_idx * 39 +: 39] <= {39{1'b0}};
		    end else if(i_in_valid)begin
		        r_mul[mul_idx * 39 +: 39] <= mul[mul_idx * 39 +: 39];
		    end
		end
	end
endgenerate

// bias 덧셈 +1bit 한 이유 덧셈 시 큰 bit + 1만 됨 39+6이 아님을 확인
reg       [`CO*40-1 : 0]    w_plus_bias;
reg       [`CO*40-1 : 0]    r_plus_bias;

integer bias_idx;
generate
	always @ (*) begin
		w_plus_bias[0 +: `CO*40]= {`CO*40{1'b0}};
		for(bias_idx =0; bias_idx < `CO; bias_idx = bias_idx +1) begin
			w_plus_bias[`CO*40 +: 40] = r_mul[bias_idx*39 +: 39] + $signed(i_cnn_bias[bias_idx*7 +: 7]); 
		end
	end
	always @(posedge clk or negedge reset_n) begin
	    if(!reset_n) begin
	        r_plus_bias[0 +: `CO*40] <= {`CO*40{1'b0}};
	    end else if(ce[LATENCY-2])begin
	        r_plus_bias[0 +: `CO*40] <= w_plus_bias[0 +: `CO*40];
	    end
	end
endgenerate

assign o_ot_valid = r_valid[LATENCY-1];
assign o_ot_bias = r_plus_bias;

endmodule
