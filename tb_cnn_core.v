`timescale 1ns / 1ps

`define    KX			5  // Number of Kernel X
`define    KY			5  // Number of Kernel Y

`define    W_BW         8  // BW of weight parameter
`define    B_BW         8  // BW of bias parameter
`define    M_BW         28 // I_F_BW * W_BW

`define    AK_BW        33 // M_BW + log(KY*KX) Accum Kernel 
`define    ACI_BW		35 // AK_BW + log (CI) Accum Channel Input


`define    AB_BW        35 // ACI_BW + bias (#1). 
`define    O_F_BW       35 // reLU Activation, after Activation it becomes O_F_BW-1 bit

// `define    O_F_ACC_BW   27 // for demo, O_F_BW + log (CO)

//pooling interface
`define    ST2_Pool_IBW    20 // Stage2 Pooling input bitwidth
`define    ST2_Pool_CI     3  // Number of Stage2 Pooling Channel Input
`define    ST2_Pool_CO     3  // Number of Stage2 Pooling Channel Output
`define    ST2_Pool_X      24 // Number of X (Input Channel)
`define    ST2_Pool_Y      24 // Number of y (Input Channel)

//convolution interface
`define    ST2_Conv_IBW    20 // Conv Input Bitwidth
`define    ST2_Conv_CI     3  // Number of Stage2 Conv Channel Input
`define    ST2_Conv_CO     3  // Number of Stage2 Conv Channel Output
`define    ST2_Conv_X      12 // Number of X (Conv Input Channel)
`define    ST2_Conv_Y      12 // Number of y (Con Input Channel)

module tb_cnn_core();


    reg                                                                  clk         	;
    reg                                                                  reset_n     	;
    reg                                                                  i_in_valid  	; 
    reg    signed [`ST2_Conv_CI * `ST2_Conv_IBW-1 : 0]  	             i_in_fmap    	;//3*( bitwidh) , 3ch에 대한 1point output
    wire                                                                 o_ot_valid  	;
    wire   signed [`ST2_Conv_CO * (`O_F_BW-1)-1 : 0]  		             o_ot_fmap       ;

    stage2_conv dut(
        .clk          (clk),   
        .reset_n      (reset_n),   
        .i_in_valid   (i_in_valid),   
        .i_in_fmap    (i_in_fmap),   
        .o_ot_valid   (o_ot_valid),   
        .o_ot_fmap    (o_ot_fmap)        
    );

 always #5 clk = ~clk;
    integer i,j, k;
    initial begin
        clk = 0;
        reset_n = 1;
        i_in_valid = 0;
        i_in_fmap = 0;
        #10;
        reset_n = 0;
        #10;
        reset_n = 1;
        #25;
        i_in_valid = 1;

        for (i = 0; i < `ST2_Conv_X*`ST2_Conv_Y; i = i + 1) begin
            for ( k= 0; k < `ST2_Conv_CI ; k=k+1)begin
                i_in_fmap[k*`ST2_Conv_IBW +: `ST2_Conv_IBW] = $signed(k+i);
            end
            #10;  
        end 
        


        for (j=0 ; j<`ST2_Conv_X * `ST2_Conv_Y; j = j + 1) begin
            wait(o_ot_valid == 1);
        end        
        #10;


        #100;


        $finish;
    end

endmodule
