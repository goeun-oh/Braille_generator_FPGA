
`timescale 1ns / 1ps
`include "stage2_defines_cnn_core.v"

module stage2_cnn_kernel (
    // Clock & Reset
input                               		   clk         	,
input                               		   reset_n     	,

//5x5x7
input     signed [`KX*`KY*`W_BW-1 : 0] 	       i_cnn_weight ,
input                                          i_in_valid  	,
input     signed [`KX*`KY*`ST2_Conv_IBW-1 : 0] i_in_fmap    , //5x5x(20bit)
output                                         o_ot_valid  	,
output    signed [`AK_BW-1 : 0]  			   o_ot_kernel_acc           
    );

localparam LATENCY = 2;


//==============================================================================
// Data Enable Signals 
//==============================================================================
wire    [LATENCY-1 : 0] 	ce;
reg     [LATENCY-1 : 0] 	r_valid;
always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_valid   <= 0;
    end else begin
        r_valid[LATENCY-2]  <= i_in_valid;
        r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
    end
end

assign	ce = r_valid;

//==============================================================================
// mul = fmap * weight
//==============================================================================

wire      signed [`KY*`KX*`M_BW-1 : 0]    mul  ;
//5x5 28bit
reg       signed [`KY*`KX*`M_BW-1 : 0]    r_mul;



genvar mul_idx;
generate
	//커널사이즈(5x5만큼 한번의 곱셈하기 위함)
	for(mul_idx = 0; mul_idx < `KY*`KX; mul_idx = mul_idx + 1) begin : gen_mul
		assign  mul[mul_idx * `M_BW +: `M_BW]	=  $signed(i_in_fmap[mul_idx * `ST2_Conv_IBW +: `ST2_Conv_IBW]) *  $signed(i_cnn_weight[mul_idx * `W_BW +: `W_BW]);
	
		always @(posedge clk or negedge reset_n) begin
		    if(!reset_n) begin
		        r_mul[mul_idx * `M_BW +: `M_BW] <= 0;
		    end else if(i_in_valid)begin
		        r_mul[mul_idx * `M_BW +: `M_BW] <= $signed(mul[mul_idx * `M_BW +: `M_BW]);
				
		    end
		end
	end
endgenerate

    //debug
    reg signed [`M_BW-1:0] d_mul [0:`KY-1][0:`KX-1];    
integer j, i;
		always @(posedge clk or negedge reset_n) begin
		    if(!reset_n) begin
				for(j=0;j<`KY;j=j+1)begin
					for(i=0; i<`KX;i=i+1) begin
						d_mul[j][i]<=0;
					end
				end
		    end else if(i_in_valid)begin
				for(j=0;j<`KY;j=j+1)begin
					for(i=0; i<`KX;i=i+1) begin
						d_mul[j][i]<=mul[(j*`KX+i) * `M_BW +: `M_BW];
					end
				end	
		    end
		end

reg       signed [`AK_BW-1 : 0]    acc_kernel 	;
reg       signed [`AK_BW-1 : 0]    r_acc_kernel   ;


//25개 accumulate
integer acc_idx;
generate
	always @ (*) begin
		acc_kernel[0 +: `AK_BW]= 0;
		for(acc_idx =0; acc_idx < `KY*`KX; acc_idx = acc_idx +1) begin
			acc_kernel[0 +: `AK_BW] = $signed(acc_kernel[0 +: `AK_BW]) + $signed(r_mul[acc_idx*`M_BW +: `M_BW]); 
		end
	end
	always @(posedge clk or negedge reset_n) begin
	    if(!reset_n) begin
	        r_acc_kernel[0 +: `AK_BW] <= 0;
	    end else if(ce[LATENCY-2])begin
	        r_acc_kernel[0 +: `AK_BW] <= $signed(acc_kernel[0 +: `AK_BW]);
	    end
	end
endgenerate

assign o_ot_valid = r_valid[LATENCY-1];
assign o_ot_kernel_acc = r_acc_kernel;

endmodule