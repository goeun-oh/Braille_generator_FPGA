
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

localparam LATENCY = 3+25;

integer i,j,c;
//==============================================================================
// Data Enable Signals 
//==============================================================================
wire    [LATENCY-1 : 0] 	ce;
reg     [LATENCY-1 : 0] 	r_valid;
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        r_valid <= 0;
    end else begin
        r_valid[0] <= i_in_valid;
        for ( i = 1; i < LATENCY; i = i + 1) begin
            r_valid[i] <= r_valid[i - 1];
        end
    end
end
assign	ce = r_valid;

//==============================================================================
// mul = fmap * weight
//==============================================================================

reg       signed [`M_BW-1 : 0]  mul [0:`KY-1][0:`KX-1];
//5x5 28bit
reg       signed [`M_BW-1 : 0]  r_mul [0:(`KY*`KX)-1][0:`KY-1][0:`KX-1];


	always @(posedge clk or negedge reset_n) begin
		if(!reset_n) begin
			for(c=0; c<`KY*`KX; c=c+1) begin
				for(j=0;j<`KY;j=j+1)begin
					for(i=0; i<`KX;i=i+1) begin
						r_mul[c][j][i] <= 0;
					end
				end
			end
		end else begin
			for(c=0; c<`KY*`KX-1; c= c+1) begin
				for(j=0;j<`KY;j=j+1)begin
					for(i=0; i<`KX;i=i+1) begin
						r_mul[c+1][j][i] <= r_mul[c][j][i];
					end
				end	
			end
		end
	end



//i_in_valid 들어오면 25개 각각 곱셈
genvar y,x;
generate
	for(y=0 ; y<`KY ; y=y+1) begin
		for(x=0 ; x<`KX ; x=x+1) begin
			(* use_dsp = "yes" *) 
			//이게 1clk안에 가능? setup, hold지카면서?
			// assign  mul[(y*`KX+x)* `M_BW +: `M_BW]	=  $signed(i_in_fmap[(y*`KX+x)* `ST2_Conv_IBW +: `ST2_Conv_IBW]) *  $signed(i_cnn_weight[(y*`KX+x) * `W_BW +: `W_BW]);
			always @(posedge clk or negedge reset_n) begin
		    	if(!reset_n) begin
					mul[y][x] <= 0;
		    	end else if(i_in_valid)begin
					mul[y][x] <= 
						$signed(i_in_fmap[(y*`KX+x)*`ST2_Conv_IBW +: `ST2_Conv_IBW]) * 
						$signed(i_cnn_weight[(y*`KX+x)*`W_BW +: `W_BW]);					
				end
			end
		end
	end	
endgenerate

//r_valid[0], r_mul[0] 
	always @(posedge clk or negedge reset_n) begin
		if(!reset_n) begin
			for(c=0; c<`KY*`KX; c=c+1) begin
				for(j=0;j<`KY;j=j+1)begin
					for(i=0; i<`KX;i=i+1) begin
						r_mul[0][j][i] <= 0;
					end
				end
			end
		end else if(r_valid[0])begin
			for(c=0; c<`KY*`KX; c= c+1) begin
				for(j=0;j<`KY;j=j+1)begin
					for(i=0; i<`KX;i=i+1) begin
						r_mul[0][j][i] <= $signed(mul[j][i]);
					end
				end	
			end
		end
	end

    //debug
    reg signed [`M_BW-1:0] d_mul [0:`KY-1][0:`KX-1];    
	always @(posedge clk or negedge reset_n) begin
		if(!reset_n) begin
			for(j=0;j<`KY;j=j+1)begin
				for(i=0; i<`KX;i=i+1) begin
					d_mul[j][i]<=0;
				end
			end
		end else if(r_valid[0])begin
			for(j=0;j<`KY;j=j+1)begin
				for(i=0; i<`KX;i=i+1) begin
					d_mul[j][i] <= $signed(mul[j][i]);
				end
			end	
		end
	end





//r_valid[1], r_mul[1]
reg       signed [`AK_BW-1 : 0]    acc_kernel[0:`KY*`KX-1]  	;
reg       signed [`AK_BW-1 : 0]    r_acc_kernel         ;
reg [4:0] acc_idx;  // 0~24 index
reg accumulating;
reg acc_done;

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
		for(c=0; c<`KX*`KY; c=c+1 ) begin
        	acc_kernel[c] <= 0;
		end
		r_acc_kernel<=0;
    end else if (r_valid[1])  begin
		acc_kernel[0] <= 	r_mul[0][0][0];
	end else if (r_valid[2]) begin
		acc_kernel[1] <= acc_kernel[0] + r_mul[1][0][1];
	end else if (r_valid[3]) begin
		acc_kernel[2] <= acc_kernel[1] + r_mul[2][0][2];
	end else if (r_valid[4]) begin
		acc_kernel[3] <= acc_kernel[2] + r_mul[3][0][3];
	end else if (r_valid[5]) begin
		acc_kernel[4] <= acc_kernel[3] + r_mul[4][0][4];
	end else if (r_valid[6]) begin
		acc_kernel[5] <= acc_kernel[4] + r_mul[5][1][0];
	end else if (r_valid[7]) begin
		acc_kernel[6] <= acc_kernel[5] + r_mul[6][1][1];
	end else if (r_valid[8]) begin
		acc_kernel[7] <= acc_kernel[6] + r_mul[7][1][2];
	end else if (r_valid[9]) begin
		acc_kernel[8] <= acc_kernel[7] + r_mul[8][1][3];
	end else if (r_valid[10]) begin
		acc_kernel[9] <= acc_kernel[8] + r_mul[9][1][4];
	end else if (r_valid[11]) begin
		acc_kernel[10] <= acc_kernel[9] + r_mul[10][2][0];
	end else if (r_valid[12]) begin
		acc_kernel[11] <= acc_kernel[10] + r_mul[11][2][1];
	end else if (r_valid[13]) begin
		acc_kernel[12] <= acc_kernel[11] + r_mul[12][2][2];
	end else if (r_valid[14]) begin
		acc_kernel[13] <= acc_kernel[12] + r_mul[13][2][3];
	end else if (r_valid[15]) begin
		acc_kernel[14] <= acc_kernel[13] + r_mul[14][2][4];
	end else if (r_valid[16]) begin
		acc_kernel[15] <= acc_kernel[14] + r_mul[15][3][0];
	end else if (r_valid[17]) begin
		acc_kernel[16] <= acc_kernel[15] + r_mul[16][3][1];
	end else if (r_valid[18]) begin
		acc_kernel[17] <= acc_kernel[16] + r_mul[17][3][2];
	end else if (r_valid[19]) begin
		acc_kernel[18] <= acc_kernel[17] + r_mul[18][3][3];
	end else if (r_valid[20]) begin
		acc_kernel[29] <= acc_kernel[18] + r_mul[19][3][4];	
	end else if (r_valid[21]) begin
		acc_kernel[20] <= acc_kernel[19] + r_mul[20][4][0];	
	end else if (r_valid[22]) begin
		acc_kernel[21] <= acc_kernel[20] + r_mul[21][4][1];	
	end else if (r_valid[23]) begin
		acc_kernel[22] <= acc_kernel[21] + r_mul[22][4][2];	
	end else if (r_valid[24]) begin
		acc_kernel[23] <= acc_kernel[22] + r_mul[23][4][3];		
	end else if (r_valid[25]) begin
		acc_kernel[24] <= acc_kernel[23] + r_mul[24][4][4];										
	end else if (r_valid[26]) begin
		for(c=0; c<`KX*`KY; c=c+1 ) begin
        	acc_kernel[c] <= 0;
		end
		r_acc_kernel <= acc_kernel[24];
	end
end


// //25개 accumulate
// // integer acc_idx;
// //=== [누산 단계 분할: partial sum 5개 생성] ===//
// wire signed [`AK_BW-1:0] partial_sum[0:4];

// genvar psum_idx;

// generate
// 	// always @ (*) begin
// 	// 	acc_kernel[0 +: `AK_BW]= 0;
// 	// 	for(acc_idx =0; acc_idx < `KY*`KX; acc_idx = acc_idx +1) begin
// 	// 		acc_kernel[0 +: `AK_BW] = $signed(acc_kernel[0 +: `AK_BW]) + $signed(r_mul[acc_idx*`M_BW +: `M_BW]); 
// 	// 	end
// 	// end
// 	// always @(posedge clk or negedge reset_n) begin
// 	//     if(!reset_n) begin
// 	//         r_acc_kernel[0 +: `AK_BW] <= 0;
// 	//     end else if(ce[LATENCY-2])begin
// 	//         r_acc_kernel[0 +: `AK_BW] <= $signed(acc_kernel[0 +: `AK_BW]);
// 	//     end
// 	// end
//     for (psum_idx = 0; psum_idx < `KX; psum_idx = psum_idx + 1) begin : gen_partial_sum
//       assign partial_sum[psum_idx] =
//         $signed(r_mul[(psum_idx*`KX + 0)*`M_BW +: `M_BW]) +
//         $signed(r_mul[(psum_idx*`KX + 1)*`M_BW +: `M_BW]) +
//         $signed(r_mul[(psum_idx*`KX + 2)*`M_BW +: `M_BW]) +
//         $signed(r_mul[(psum_idx*`KX + 3)*`M_BW +: `M_BW]) +
//         $signed(r_mul[(psum_idx*`KX + 4)*`M_BW +: `M_BW]);
//     end	
// endgenerate

// //=== [레지스터에 partial sum 저장] ===//
// reg signed [`AK_BW-1:0] r_partial_sum[0:4];

// always @(posedge clk or negedge reset_n) begin
//   if (!reset_n) begin
//     for (i = 0; i < 5; i = i + 1)
//       r_partial_sum[i] <= 0;
//   end else if (ce[LATENCY-3]) begin
//     for (i = 0; i < 5; i = i + 1)
//       r_partial_sum[i] <= partial_sum[i];
//   end
// end



//1clk
// always @(posedge clk or negedge reset_n) begin
//   if (!reset_n)
//     r_acc_kernel <= 0;
//   else if (ce[LATENCY-2])
//     r_acc_kernel <= r_partial_sum[0] + r_partial_sum[1] +
//                     r_partial_sum[2] + r_partial_sum[3] + r_partial_sum[4];
// end

assign o_ot_valid = r_valid[27];
assign o_ot_kernel_acc = r_acc_kernel;

endmodule