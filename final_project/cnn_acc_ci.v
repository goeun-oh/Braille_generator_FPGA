/*******************************************************************************
Author: joohan.kim (https://blog.naver.com/chacagea)
Asso`ciated Filename: cnn_core.v
Purpose: verilog code to understand the CNN operation
License : https://github.com/matbi86/matbi_fpga_season_1/blob/main/LICENSE
Revision History: February 13, 2020 - initial release
*******************************************************************************/

`include "timescale.v"
`include "defines_cnn_core.v"
module cnn_acc_ci (
    // Clock & Reset
input                                   clk         ,
input                                   reset_n     ,

//3*5*5*(7)
input     [`CI*`KX*`KY*`W_BW-1 : 0]  	i_cnn_weight,
input                                   i_in_valid  ,
input     [`CI*`KX*`KY*`ST2_Conv_IBW-1 : 0]  	i_in_fmap   ,
output                                  o_ot_valid  ,
output    [`ACI_BW-1 : 0]  			    o_ot_ci_acc 	     
    );

localparam LATENCY = 1;


//==============================================================================
// Data Enable Signals 
//==============================================================================
wire    [LATENCY-1 : 0] 	ce;
reg     [LATENCY-1 : 0] 	r_valid;
wire    [`CI-1 : 0]          w_ot_valid;
always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_valid   <= {LATENCY{1'b0}};
    end else begin
        r_valid[LATENCY-1]  <= &w_ot_valid;
    end
end

assign	ce = r_valid;
//==============================================================================
// mul_acc kenel instance
//==============================================================================

wire    [`CI-1 : 0]             w_in_valid;
wire    [`CI*`AK_BW-1 : 0]  	w_ot_kernel_acc;
wire    [`ACI_BW-1 : 0]  		w_ot_ci_acc;
reg     [`ACI_BW-1 : 0]  		r_ot_ci_acc;

genvar mul_inst;
generate
	for(mul_inst = 0; mul_inst < `CI; mul_inst = mul_inst + 1) begin : gen_mul_inst
		wire    [`KX*`KY*`W_BW-1 : 0]  	w_cnn_weight 	= i_cnn_weight[mul_inst*`KY*`KX*`W_BW +: `KY*`KX*`W_BW];
		wire    [`KX*`KY*`ST2_Conv_IBW-1 : 0]  	w_in_fmap    	= i_in_fmap[mul_inst*`KY*`KX*`ST2_Conv_IBW +: `KY*`KX*`ST2_Conv_IBW];
		assign	w_in_valid[mul_inst] = i_in_valid; 
		cnn_kernel u_cnn_kernel(
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

	reg    [`ACI_BW-1 : 0]  		ot_ci_acc;
	integer i;
	always @(*) begin
		ot_ci_acc = {`ACI_BW{1'b0}};
		for(i = 0; i < `CI; i = i+1) begin
			ot_ci_acc = ot_ci_acc + w_ot_kernel_acc[i*`AK_BW +: `AK_BW];
		end
	end

//assign w_ot_ci_acc = w_ot_kernel_acc[0*`AK_BW +: `AK_BW] + w_ot_kernel_acc[(0+1)*`AK_BW +: `AK_BW];
assign w_ot_ci_acc = ot_ci_acc;

always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_ot_ci_acc[0 +: `ACI_BW] <= {`ACI_BW{1'b0}};
    end else if(&w_ot_valid)begin
        r_ot_ci_acc[0 +: `ACI_BW] <= w_ot_ci_acc[0 +: `ACI_BW];
    end
end

assign o_ot_valid = r_valid[LATENCY-1];
assign o_ot_ci_acc = r_ot_ci_acc;

endmodule
