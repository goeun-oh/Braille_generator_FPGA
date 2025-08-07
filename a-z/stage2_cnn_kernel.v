
// `timescale 1ns / 1ps

// `include "defines_cnn_core.v"

// module stage2_cnn_kernel (
//     // Clock & Reset
// input                               		   clk         	,
// input                               		   reset_n     	,

// //5x5x7
// input     signed [`KX*`KY*`ST2_W_BW-1 : 0] 	       i_cnn_weight ,
// input                                              i_in_valid  	,
// input     signed [`KX*`KY*`ST2_Conv_IBW-1 : 0]     i_in_fmap    , //5x5x(20bit)
// output                                             o_ot_valid  	,
// output    signed [`ST2_AK_BW-1 : 0]  			   o_ot_kernel_acc           
//     );

// localparam LATENCY = 3+25;

// integer i,j,k,c;
// //==============================================================================
// // Data Enable Signals 
// //==============================================================================
// wire    [LATENCY-1 : 0] 	ce;
// reg     [LATENCY-1 : 0] 	r_valid;
// always @(posedge clk or negedge reset_n) begin
//     if (!reset_n) begin
//         r_valid <= 0;
//     end else begin
//         r_valid[0] <= i_in_valid;
//         for ( i = 1; i < LATENCY; i = i + 1) begin
//             r_valid[i] <= r_valid[i - 1];
//         end
//     end
// end
// assign	ce = r_valid;

// //==============================================================================
// // mul = fmap * weight
// //==============================================================================

// reg       signed [`ST2_M_BW-1 : 0]  mul [0:`KY-1][0:`KX-1];
// //5x5 28bit
// reg       signed [`ST2_M_BW-1 : 0]  r_mul [0:(`KY*`KX)-1][0:`KY-1][0:`KX-1];


// 	//i_in_valid 들어오면 25개 각각 곱셈
// 	genvar y,x;
// 	generate
// 		for(y=0 ; y<`KY ; y=y+1) begin
// 			for(x=0 ; x<`KX ; x=x+1) begin
// 				(* use_dsp = "yes" *) 
// 				//이게 1clk안에 가능? setup, hold지카면서?
// 				// assign  mul[(y*`KX+x)* `ST2_M_BW +: `ST2_M_BW]	=  $signed(i_in_fmap[(y*`KX+x)* `ST2_Conv_IBW +: `ST2_Conv_IBW]) *  $signed(i_cnn_weight[(y*`KX+x) * `ST2_W_BW +: `ST2_W_BW]);
// 				always @(posedge clk or negedge reset_n) begin
// 					if(!reset_n) begin
// 						mul[y][x] <= 0;
// 					end else if(i_in_valid)begin
// 						mul[y][x] <= 
// 							$signed(i_in_fmap[(y*`KX+x)*`ST2_Conv_IBW +: `ST2_Conv_IBW]) * 
// 							$signed(i_cnn_weight[(y*`KX+x)*`ST2_W_BW +: `ST2_W_BW]);					
// 					end
// 				end
// 			end
// 		end	
// 	endgenerate



// 	always @(posedge clk or negedge reset_n) begin
// 		if(!reset_n) begin
// 			for(c=0; c<`KY*`KX; c=c+1) begin
// 				for(j=0;j<`KY;j=j+1)begin
// 					for(i=0; i<`KX;i=i+1) begin
// 						r_mul[c][j][i] <= 0;
// 					end
// 				end
// 			end
// 		end else begin
// 			for(c=0; c<`KY*`KX-1; c= c+1) begin
// 				for(j=0;j<`KY;j=j+1)begin
// 					for(i=0; i<`KX;i=i+1) begin
// 						r_mul[c+1][j][i] <= r_mul[c][j][i];
// 					end
// 				end	
// 			end
// 			if(r_valid[0])begin
// 				for(j=0;j<`KY;j=j+1)begin
// 					for(i=0; i<`KX;i=i+1) begin
// 						r_mul[0][j][i] <= $signed(mul[j][i]);
// 					end
// 				end	
// 			end			
// 		end
// 	end

//     //debug
//     reg signed [`ST2_M_BW-1:0] d_mul [0:`KY-1][0:`KX-1];    
// 	always @(posedge clk or negedge reset_n) begin
// 		if(!reset_n) begin
// 			for(j=0;j<`KY;j=j+1)begin
// 				for(i=0; i<`KX;i=i+1) begin
// 					d_mul[j][i]<=0;
// 				end
// 			end
// 		end else if(r_valid[0])begin
// 			for(j=0;j<`KY;j=j+1)begin
// 				for(i=0; i<`KX;i=i+1) begin
// 					d_mul[j][i] <= $signed(mul[j][i]);
// 				end
// 			end	
// 		end
// 	end


// //r_valid[1], r_mul[1]
// reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel[0:`KY*`KX-1]  	;
// reg       signed [`ST2_AK_BW-1 : 0]    r_acc_kernel         ;
// reg [4:0] acc_idx;  // 0~24 index
// reg accumulating;
// reg acc_done;

// always @(posedge clk or negedge reset_n) begin
//     if (!reset_n) begin
//         for (k = 0; k < 25; k = k + 1) begin
//             acc_kernel[k] <= 0;
//         end
//         r_acc_kernel <= 0;
//     end else begin
//         for (k = 0; k < 25; k = k + 1) begin
//             if (r_valid[k + 1]) begin
//                 if (k == 0)
//                     acc_kernel[0] <= r_mul[0][0][0];
//                 else begin
//                     // y, x 위치 계산
//                     	j = k / 5;
//                     	i = k % 5;
//                     acc_kernel[k] <= acc_kernel[k - 1] + r_mul[k][j][i];
//                 end
//             end
//         end

//         // 마지막 결과 저장
//         if (r_valid[26]) begin
//             r_acc_kernel <= acc_kernel[24];
//         end
//     end
// end


// assign o_ot_valid = r_valid[27];
// assign o_ot_kernel_acc = r_acc_kernel;

// endmodule


`timescale 1ns / 1ps
`include "defines_cnn_core.v"

module stage2_cnn_kernel (
    // Clock & Reset
input                               		   clk         	,
input                               		   reset_n     	,

//5x5x7
input     signed [`KX*`KY*`ST2_W_BW-1 : 0] 	       i_cnn_weight ,
input                                          i_in_valid  	,
input     signed [`KX*`KY*`ST2_Conv_IBW-1 : 0] i_in_fmap    , //5x5x(20bit)
output                                         o_ot_valid  	,
output    signed [`ST2_AK_BW-1 : 0]  			   o_ot_kernel_acc           
    );

localparam LATENCY = 3;


//==============================================================================
// Data Enable Signals 
//==============================================================================
wire    [LATENCY-1 : 0] 	ce;
reg     [LATENCY-1 : 0] 	r_valid;
always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_valid   <= 0;
    end else begin
        r_valid[LATENCY-3]  <= i_in_valid;
        r_valid[LATENCY-2]  <= r_valid[LATENCY-3];
        r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
    end
end

//==============================================================================
// mul = fmap * weight
//==============================================================================

wire      signed [`KY*`KX*`ST2_M_BW-1 : 0]    mul  ;
//5x5 28bit
reg       signed [`KY*`KX*`ST2_M_BW-1 : 0]    r_mul;



//i_in_valid
genvar mul_idx;
generate
	//커널사이즈(5x5만큼 한번의 곱셈하기 위함)
	for(mul_idx = 0; mul_idx < `KY*`KX; mul_idx = mul_idx + 1) begin : gen_mul
		assign  mul[mul_idx * `ST2_M_BW +: `ST2_M_BW]	=  $signed(i_in_fmap[mul_idx * `ST2_Conv_IBW +: `ST2_Conv_IBW]) *  $signed(i_cnn_weight[mul_idx * `ST2_W_BW +: `ST2_W_BW]);
	
		always @(posedge clk or negedge reset_n) begin
		    if(!reset_n) begin
		        r_mul[mul_idx * `ST2_M_BW +: `ST2_M_BW] <= 0;
		    end else if(i_in_valid)begin
		        r_mul[mul_idx * `ST2_M_BW +: `ST2_M_BW] <= $signed(mul[mul_idx * `ST2_M_BW +: `ST2_M_BW]);
				
		    end
		end
	end
endgenerate

//     //debug
//     reg signed [`ST2_M_BW-1:0] d_mul [0:`KY-1][0:`KX-1];    
// integer j, i;
// 		always @(posedge clk or negedge reset_n) begin
// 		    if(!reset_n) begin
// 				for(j=0;j<`KY;j=j+1)begin
// 					for(i=0; i<`KX;i=i+1) begin
// 						d_mul[j][i]<=0;
// 					end
// 				end
// 		    end else if(i_in_valid)begin
// 				for(j=0;j<`KY;j=j+1)begin
// 					for(i=0; i<`KX;i=i+1) begin
// 						d_mul[j][i]<=mul[(j*`KX+i) * `ST2_M_BW +: `ST2_M_BW];
// 					end
// 				end	
// 		    end
// 		end

reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel_0[0:1];
reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel_1[0:1];
reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel_2[0:1];
reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel_3[0:1];
reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel_4[0:1];
reg       signed [`ST2_AK_BW-1 : 0]    r_acc_kernel;


//25개 accumulate
	integer i;
	always @ (*) begin
		acc_kernel_0[0]= 0;
		acc_kernel_1[0]= 0;
		acc_kernel_2[0]= 0;
		acc_kernel_3[0]= 0;
		acc_kernel_4[0]= 0;
		for(i =0; i < `KX; i = i +1) begin
			acc_kernel_0[0] = acc_kernel_0[0] + $signed(r_mul[i*`ST2_M_BW            +: `ST2_M_BW]); 
			acc_kernel_1[0] = acc_kernel_1[0] + $signed(r_mul[(i+1*`KX)*`ST2_M_BW    +: `ST2_M_BW]); 
			acc_kernel_2[0] = acc_kernel_2[0] + $signed(r_mul[(i+2*`KX)*`ST2_M_BW    +: `ST2_M_BW]); 
			acc_kernel_3[0] = acc_kernel_3[0] + $signed(r_mul[(i+3*`KX)*`ST2_M_BW    +: `ST2_M_BW]); 
			acc_kernel_4[0] = acc_kernel_4[0] + $signed(r_mul[(i+4*`KX)*`ST2_M_BW    +: `ST2_M_BW]); 
		end
	end


//r_valid[0]
	always @(posedge clk or negedge reset_n) begin
	    if(!reset_n) begin
	        acc_kernel_0[1] <= 0;
			acc_kernel_1[1] <= 0;
			acc_kernel_2[1] <= 0;
			acc_kernel_3[1] <= 0;
			acc_kernel_4[1] <= 0;
	    end else if(r_valid[LATENCY-3])begin
			acc_kernel_0[1] <= acc_kernel_0[0];
			acc_kernel_1[1] <= acc_kernel_1[0];
			acc_kernel_2[1] <= acc_kernel_2[0];
			acc_kernel_3[1] <= acc_kernel_3[0];
			acc_kernel_4[1] <= acc_kernel_4[0];
	    end
	end



//r_valid[1]
	always @(posedge clk or negedge reset_n) begin
	    if(!reset_n) begin
			r_acc_kernel <= 0;
	    end else if(r_valid[LATENCY-2])begin
			r_acc_kernel <= acc_kernel_0[1]+acc_kernel_1[1]+acc_kernel_2[1]+acc_kernel_3[1]+acc_kernel_4[1];
	    end
	end


//r_valid[2]
assign o_ot_valid = r_valid[LATENCY-1];
assign o_ot_kernel_acc = r_acc_kernel;

endmodule
