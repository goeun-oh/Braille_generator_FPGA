`timescale 1ns / 1ps

`include "defines_cnn_core.v"

module stage2_pooling_core(
    input                                                                                   clk         	,
    input                                                                                   reset_n     	,
    input                                                                                   i_in_valid  	,
    input         [`ST2_Pool_CI * `ST2_Pool_IBW - 1 : 0]                                    i_in_fmap    	,//3*(19bit) , 3ch에 대한 1point input
    output                                                                                  o_ot_valid  	,
    output reg    [`ST2_Pool_CI * `ST2_Conv_IBW - 1 : 0]                                    o_ot_fmap        //3*(19bit) , 3ch에 대한 1point output
    );

    localparam LATENCY = 1;


    wire    [LATENCY-1 : 0] 	                   ce;
    reg     [LATENCY-1 : 0] 	                   r_valid;

    wire    [`ST2_Pool_CI-1 : 0]                   w_in_valid; //3bit valid, each bit(LSB to MSB) is used in each instance
    wire    [`ST2_Pool_CO * `ST2_Pool_IBW - 1 : 0] w_ot_fmap;  //3*19bit
    wire    [`ST2_Pool_CO-1 : 0]                   w_ot_valid; 
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid   <= {LATENCY{1'b0}};
        end else begin
            r_valid[LATENCY-1]  <= &w_ot_valid;
        end
    end

assign	ce = r_valid;

    genvar ci_inst;
    generate
        for(ci_inst = 0; ci_inst < `ST2_Pool_CI; ci_inst = ci_inst + 1) begin 

            wire [`ST2_Pool_IBW - 1 : 0] w_in_fmap;
            
            assign w_in_fmap = i_in_fmap[ci_inst * `ST2_Pool_IBW +: `ST2_Pool_IBW];
                
            assign w_in_valid[ci_inst] = i_in_valid; 

            stage2_pooling u_stage2_pooling(
                .clk         	(clk),
                .reset_n     	(reset_n),
                .i_in_valid  	(w_in_valid[ci_inst]),
                .i_in_fmap    	(w_in_fmap),

                //w_ot_valid 3비트 중 LSB부터 MSB로 하나씩 생성된 인스턴스 3개로부터 valid신호를 받음
                .o_ot_valid  	(w_ot_valid[ci_inst]),

                //3개의 인스턴스로부터 받은 output을 일렬로 저장
                .o_ot_fmap      (w_ot_fmap[ci_inst * `ST2_Pool_IBW +: `ST2_Pool_IBW])
                );  
        end
    endgenerate

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            o_ot_fmap <= 0;
        end else if(&w_ot_valid) begin
            o_ot_fmap <= w_ot_fmap;
        end
    end
    
    assign o_ot_valid = r_valid[LATENCY-1];
  
// debug
reg signed [`ST2_Pool_IBW - 1 : 0] d_i_in_fmap [0 : `ST2_Pool_CI - 1];
integer ch;
   always @(posedge clk or negedge reset_n) begin
       if(!reset_n) begin
           for(ch =0 ; ch<`ST2_Pool_CI; ch=ch+1) begin
               d_i_in_fmap [ch] <= 0;
           end
       end else if(i_in_valid) begin
           for(ch =0 ; ch<`ST2_Pool_CI; ch=ch+1) begin
               d_i_in_fmap [ch] <= $signed(i_in_fmap[ch*`ST2_Pool_IBW+:`ST2_Pool_IBW]);
           end
       end
   end
    


    endmodule