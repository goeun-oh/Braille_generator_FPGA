
`timescale 1ns / 1ps
`include "stage2_defines_cnn_core.v"
module stage2_cnn_acc_ci (
    // Clock & Reset
input                                           		clk         ,
input                                           		reset_n     ,

//3*5*5*(7)
input     signed [`ST2_Conv_CI*`KX*`KY*`W_BW-1 : 0]  			i_cnn_weight,
input                                           				i_in_valid  ,
input     signed [`ST2_Conv_CI*`KX*`KY*`ST2_Conv_IBW-1 : 0]  	i_in_fmap   , //3x5x5x(20bit)
output                                          				o_ot_valid  ,
output    signed [`ACI_BW-1 : 0]  			            		o_ot_ci_acc 	     
    );

localparam LATENCY = 1;


//==============================================================================
// Data Enable Signals 
//==============================================================================
wire    [LATENCY-1 : 0] 	ce;
reg     [LATENCY-1 : 0] 	r_valid;
wire    [`ST2_Conv_CI-1 : 0]          w_ot_valid;
always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_valid   <= 0;
    end else begin
        r_valid[LATENCY-1]  <= &w_ot_valid;
    end
end

assign	ce = r_valid;
//==============================================================================
// mul_acc kenel instance
//==============================================================================

wire    	   [`ST2_Conv_CI-1 : 0]                 w_in_valid;
wire    signed [`ST2_Conv_CI*`AK_BW-1 : 0]  	    w_ot_kernel_acc;
wire    signed [`ACI_BW-1 : 0]  					w_ot_ci_acc;
reg     signed [`ACI_BW-1 : 0]  					r_ot_ci_acc;

genvar mul_inst;
generate
	for(mul_inst = 0; mul_inst < `ST2_Conv_CI; mul_inst = mul_inst + 1) begin : gen_mul_inst
		wire    signed [`KX*`KY*`W_BW-1 : 0]  			w_cnn_weight 	= $signed(i_cnn_weight[mul_inst*`KY*`KX*`W_BW +: `KY*`KX*`W_BW]);
		wire    signed [`KX*`KY*`ST2_Conv_IBW-1 : 0]  	w_in_fmap    	= $signed(i_in_fmap[mul_inst*`KY*`KX*`ST2_Conv_IBW +: `KY*`KX*`ST2_Conv_IBW]);
		assign	w_in_valid[mul_inst] = i_in_valid; 
		stage2_cnn_kernel u_stage2_cnn_kernel(
    	.clk             (clk            ),
    	.reset_n         (reset_n        ),
    	.i_cnn_weight    (w_cnn_weight   ),
    	.i_in_valid      (w_in_valid[mul_inst]),
    	.i_in_fmap       (w_in_fmap      ),
    	.o_ot_valid      (w_ot_valid[mul_inst]),
    	.o_ot_kernel_acc (w_ot_kernel_acc[mul_inst*`AK_BW +: `AK_BW])             
    	);
	end
endgenerate

	reg   signed [`ACI_BW-1 : 0]  		ot_ci_acc;
	integer i;

	//@accumulator
	always @(*) begin
		ot_ci_acc = 0;
		for(i = 0; i < `ST2_Conv_CI; i = i+1) begin
			ot_ci_acc = ot_ci_acc + $signed(w_ot_kernel_acc[i*`AK_BW +: `AK_BW]);
		end
	end

//assign w_ot_ci_acc = w_ot_kernel_acc[0*`AK_BW +: `AK_BW] + w_ot_kernel_acc[(0+1)*`AK_BW +: `AK_BW];

assign w_ot_ci_acc = ot_ci_acc;

always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_ot_ci_acc[0 +: `ACI_BW] <= 0;
    end else if(&w_ot_valid)begin
        r_ot_ci_acc[0 +: `ACI_BW] <= $signed(w_ot_ci_acc[0 +: `ACI_BW]);
    end
end

assign o_ot_valid = r_valid[LATENCY-1];
assign o_ot_ci_acc = r_ot_ci_acc;

endmodule
