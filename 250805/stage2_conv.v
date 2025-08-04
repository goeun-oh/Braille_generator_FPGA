`timescale 1ns / 1ps


`include "defines_cnn_core.v"

module stage2_conv(
    // Clock & Reset
    clk             ,
    reset_n         ,
    i_in_valid      ,
    i_in_fmap       ,
    o_ot_valid      ,
    o_ot_fmap       
    );


//==============================================================================
// Input/Output declaration
//==============================================================================
input                                                                 clk         	;
input                                                                 reset_n     	;
input                                                                 i_in_valid  	;
input     signed [`ST2_Conv_CI * `ST2_Conv_IBW-1 : 0]  	              i_in_fmap    	;//3*(n bit) , 3ch에 대한 1point input
output                                                                o_ot_valid  	;
output    signed [`ST2_Conv_CO * (`ST2_O_F_BW)-1 : 0]  		          o_ot_fmap     ;//3*(n bit) , 3ch에 대한 1point output


    // 3 * (3 * 5 * 5) * (8bit)
    wire signed  [`ST2_Conv_CO*`ST2_Conv_CI*  `KX*`KY  *`ST2_W_BW -1 : 0] w_cnn_weight;
    wire signed  [`ST2_Conv_CO*`ST2_B_BW - 1  : 0]   w_cnn_bias;

    conv2_weight_rom u_weight_rom (
        .weight(w_cnn_weight) // 3x(3x5x5)
    );

    conv2_bias_rom u_bias_rom (
        .bias(w_cnn_bias) // 3x(3x5x5)
    );

    stage2_cnn_core u_stage2_cnn_core(
        .clk(clk)                     ,
        .reset_n(reset_n)             ,
        .i_cnn_weight(w_cnn_weight)   ,
        .i_cnn_bias(w_cnn_bias)                 ,
        .i_in_valid(i_in_valid)       ,
        .i_in_fmap(i_in_fmap)         ,
        .o_ot_valid(o_ot_valid)       ,
        .o_ot_fmap(o_ot_fmap)      
    );


    //debug
    reg signed [`ST2_O_F_BW-1:0] d_ot_fmap [0:`ST2_Conv_CO-1];
    integer c;
    always @(posedge clk, negedge reset_n) begin
        if (!reset_n) begin
            for(c=0 ; c< `ST2_Conv_CO; c= c+1) begin
                d_ot_fmap[c] <= 0;
            end            
        end else if (o_ot_valid) begin
            for(c=0 ; c< `ST2_Conv_CO; c= c+1) begin
                d_ot_fmap[c] <= $signed(o_ot_fmap[c*(`ST2_O_F_BW) +: (`ST2_O_F_BW)]);
            end
        end
        
    end

endmodule
