`timescale 1ns / 1ps
`include "defines_cnn_core.v"

module stage2_cnn_kernel (
    // Clock & Reset
input                                        clk            ,
input                                        reset_n        ,

//5x5x7
input     signed [`KX*`KY*`ST2_W_BW-1 : 0]           i_cnn_weight ,
input                                          i_in_valid     ,
input     signed [`KX*`KY*`ST2_Conv_IBW-1 : 0] i_in_fmap    , //5x5x(20bit)
output                                         o_ot_valid     ,
output    signed [`ST2_AK_BW-1 : 0]              o_ot_kernel_acc           
    );

localparam LATENCY = 2;


//==============================================================================
// Data Enable Signals 
//==============================================================================
wire    [LATENCY-1 : 0]    ce;
reg     [LATENCY-1 : 0]    r_valid;
always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_valid   <= 0;
    end else begin
        r_valid[LATENCY-2]  <= i_in_valid;
        r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
    end
end

assign   ce = r_valid;

//==============================================================================
// mul = fmap * weight
//==============================================================================

wire      signed [`KY*`KX*`ST2_M_BW-1 : 0]    mul  ;
//5x5 28bit
reg       signed [`KY*`KX*`ST2_M_BW-1 : 0]    r_mul;



genvar mul_idx;
generate
   //커널사이즈(5x5만큼 한번의 곱셈하기 위함)
   for(mul_idx = 0; mul_idx < `KY*`KX; mul_idx = mul_idx + 1) begin : gen_mul
      assign  mul[mul_idx * `ST2_M_BW +: `ST2_M_BW]   =  $signed(i_in_fmap[mul_idx * `ST2_Conv_IBW +: `ST2_Conv_IBW]) *  $signed(i_cnn_weight[mul_idx * `ST2_W_BW +: `ST2_W_BW]);
   
      always @(posedge clk or negedge reset_n) begin
          if(!reset_n) begin
              r_mul[mul_idx * `ST2_M_BW +: `ST2_M_BW] <= 0;
          end else if(i_in_valid)begin
              r_mul[mul_idx * `ST2_M_BW +: `ST2_M_BW] <= $signed(mul[mul_idx * `ST2_M_BW +: `ST2_M_BW]);
            
          end
      end
   end
endgenerate

    //debug
    reg signed [`ST2_M_BW-1:0] d_mul [0:`KY-1][0:`KX-1];    
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
                  d_mul[j][i]<=mul[(j*`KX+i) * `ST2_M_BW +: `ST2_M_BW];
               end
            end   
          end
      end

reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel_0;
reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel_1;
reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel_2;
reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel_3;
reg       signed [`ST2_AK_BW-1 : 0]    acc_kernel_4;
reg       signed [`ST2_AK_BW-1 : 0]    r_acc_kernel;


//25개 accumulate

   always @ (*) begin
      acc_kernel_0= 0;
      acc_kernel_1= 0;
      acc_kernel_2= 0;
      acc_kernel_3= 0;
      acc_kernel_4= 0;
      for(i =0; i < `KX; i = i +1) begin
         acc_kernel_0 = acc_kernel_0 + $signed(r_mul[i*`ST2_M_BW            +: `ST2_M_BW]); 
         acc_kernel_1 = acc_kernel_1 + $signed(r_mul[(i+1*`KX)*`ST2_M_BW    +: `ST2_M_BW]); 
         acc_kernel_2 = acc_kernel_2 + $signed(r_mul[(i+2*`KX)*`ST2_M_BW    +: `ST2_M_BW]); 
         acc_kernel_3 = acc_kernel_3 + $signed(r_mul[(i+3*`KX)*`ST2_M_BW    +: `ST2_M_BW]); 
         acc_kernel_4 = acc_kernel_4 + $signed(r_mul[(i+4*`KX)*`ST2_M_BW    +: `ST2_M_BW]); 
      end
   end
   always @(posedge clk or negedge reset_n) begin
       if(!reset_n) begin
           r_acc_kernel <= 0;
       end else if(ce[LATENCY-2])begin
           r_acc_kernel <= (acc_kernel_0 + acc_kernel_1 + acc_kernel_2 + acc_kernel_3 +acc_kernel_4);
       end
   end


assign o_ot_valid = r_valid[LATENCY-1];
assign o_ot_kernel_acc = r_acc_kernel;

endmodule

