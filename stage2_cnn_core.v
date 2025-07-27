
`timescale 1ns / 1ps
`include "stage2_defines_cnn_core.v"

module stage2_cnn_core (
    // Clock & Reset
    clk             ,
    reset_n         ,
    i_cnn_weight    ,
    i_cnn_bias      ,
    i_in_valid      ,
    i_in_fmap       ,
    o_ot_valid      ,
    o_ot_fmap             
    );
//==============================================================================
// Input/Output declaration
//==============================================================================
input                                                                   clk         	;
input                                                                   reset_n     	;
input     signed [`ST2_Conv_CI* `ST2_Conv_CO*  `KX*`KY  *`W_BW -1 : 0]  i_cnn_weight    ; // 3 * (3 * 5 * 5) * (bitwidth)
input     signed [`ST2_Conv_CI*`B_BW - 1  : 0]                          i_cnn_bias;
input                                                                   i_in_valid  	; 
input     signed [`ST2_Conv_CI * `ST2_Conv_IBW-1 : 0]  	                i_in_fmap    	;//3*( bitwidh) , 3ch에 대한 1point output
output                                                                  o_ot_valid  	;
output    signed [`ST2_Conv_CO * (`O_F_BW-1)-1 : 0]  		            o_ot_fmap       ;//3*( bitwidh)    

localparam LATENCY = 2;
localparam COL = `ST2_Conv_X; //12
localparam ROW = `ST2_Conv_Y; //12
    
//==============================================================================
// row,col_counter
//==============================================================================
    reg [$clog2(ROW)-1:0] row;
    reg [$clog2(COL)-1:0] col;
    reg frame_flag;
    reg col_flag;


    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            row <= 0;
            col <= 0;  
        end else if(i_in_valid) begin
            if(col == COL-1) begin
                col <= 0;
                if (row == ROW -1) begin
                    row <= 0 ;
                end else begin
                    row <= row + 1;
                end
            end else begin
                col <= col + 1;
            end
        end 
    end


    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            col_flag <=0;
            frame_flag <=0;
        end else begin
            if(col == COL-1 && i_in_valid) begin
                col_flag <= 1;
                if (row == ROW -1) begin
                    frame_flag <= 1;
                end else begin
                    frame_flag <= 0;
                end
            end else begin
                col_flag <= 0;
                frame_flag <= 0;
            end
        end
    end

//==============================================================================
// Line Buffer & 5x5 window register
//==============================================================================

    //(20bit)  3channel 5x24 line_buffer
    reg signed [`ST2_Conv_IBW-1:0] line_buffer [0:`ST2_Conv_CI-1][0:`KY-1][0:`ST2_Conv_X-1];

    //(20bit)  3channel 5x5  window
    reg signed [`ST2_Conv_IBW * `ST2_Conv_CI * `KY*`KX-1:0] window;

    integer k;
    integer j;
    integer i;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (k = 0; k < `ST2_Conv_CI; k = k + 1)
                for (j = 0; j < `KY; j = j + 1)
                    for (i = 0; i < `ST2_Conv_X; i = i + 1)
                        line_buffer[k][j][i] <= 0;
        end else if (i_in_valid) begin
            for (k = 0; k < `ST2_Conv_CI; k = k + 1) begin
                for (j = 0; j < `KY - 1; j = j + 1) begin
                    line_buffer[k][j][col] <= line_buffer[k][j + 1][col];
                end
                // 마지막 라인 (최상단)에 새로운 입력 넣기
                line_buffer[k][`KY-1][col] <= i_in_fmap[k*`ST2_Conv_IBW +: `ST2_Conv_IBW];
            end
        end
    end


//==============================================================================
// receive 1px data to 3ch Line Buffer
//==============================================================================


    
//==============================================================================
// allocate data from line buffer to window, send valid signal
//==============================================================================
//                   3 * 5 * 5 * (19bit)
// reg signed [`ST2_Conv_CI * `KY*`KX * `ST2_Conv_IBW-1:0] window;

localparam V_LATENCY = 1;
reg w_valid;
reg [V_LATENCY-1 : 0] 	r_w_valid;

    //debug
    reg signed [`ST2_Conv_IBW-1:0] d_window [0:`ST2_Conv_CI-1][0:`KY-1][0:`KX-1];    

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            window <= 0;
            for (k = 0; k < `ST2_Conv_CI; k=k+1) begin
                for (j = 0; j < `KY; j=j+1) begin
                    for (i = 0; i < `KX; i = i + 1) begin
                        d_window[k][j][i] <= 0;
                    end
                end
            end
        end else if( row>=4 && col >=5 ) begin
            for (k = 0; k < `ST2_Conv_CI; k=k+1) begin
                for (j = 0; j < `KY; j=j+1) begin
                    for (i = 0; i < `KX; i = i + 1) begin
                        window[((k*`KY*`KX) + (j*`KX) + i)* `ST2_Conv_IBW +: `ST2_Conv_IBW] 
                            <= line_buffer[k][j][col-5+i];
                        // debug
                        d_window[k][j][i] <= line_buffer[k][j][col-5+i];
                    end
                end
            end
        end else if ( (row>=5 && !col) | frame_flag ) begin
            for (k = 0; k < `ST2_Conv_CI; k=k+1) begin
                for (j = 0; j < `KY; j=j+1) begin
                    for (i = 0; i < `KX; i = i + 1) begin
                        window[((k*`KY*`KX) + (j*`KX) + i)* `ST2_Conv_IBW +: `ST2_Conv_IBW] 
                            <= line_buffer[k][j][`ST2_Conv_X-`KX + i]; // col 7,8,9,10,11
                        //debug
                        d_window[k][j][i] <= line_buffer[k][j][`ST2_Conv_X-`KX + i];
                    end
                end
            end
        end 
    end

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            w_valid <= 0;
        end else if(i_in_valid &&(row>=4) && (col>=4)) begin
            w_valid <= 1; 
        end else begin
            w_valid <= 0;
        end
    end    

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_w_valid <= 0;
        end else begin
            r_w_valid [V_LATENCY-1 : 0] <= w_valid;
        end
    end    



//==============================================================================
// Data Enable Signals 
//==============================================================================
wire    [LATENCY-1 : 0] 	          ce;
reg     [LATENCY-1 : 0] 	          r_valid;
wire    [`ST2_Conv_CO-1 : 0]          w_ot_valid;
always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_valid   <= 0;
    end else begin
        r_valid[LATENCY-2]  <= &w_ot_valid;
        r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
    end
end

assign	ce = r_valid;

//==============================================================================
// acc instance
//==============================================================================

        //valid신호 3개(3bit)
wire    [`ST2_Conv_CO-1 : 0]             w_in_valid;
        //channel 3개, 33bit
wire    signed [`ST2_Conv_CO*(`ACI_BW)-1 : 0]  w_ot_ci_acc;

wire    signed [`ST2_Conv_CI*`ST2_Conv_CO*  `KX*`KY  *`W_BW -1 : 0] w_cnn_weight;
assign w_cnn_weight = i_cnn_weight;
        //debug
reg     signed [`W_BW-1:0] d_weight [0:`ST2_Conv_CO-1][0:`ST2_Conv_CI-1][0:`KY-1][0:`KX-1];
    integer c,z,y,x;

    always @(posedge clk) begin
        for (c = 0; c < `ST2_Conv_CO; c = c + 1)
            for (z = 0; z < `ST2_Conv_CI; z = z + 1)
                for (y = 0; y < `KY; y = y + 1)
                    for (x = 0; x < `KX; x = x + 1)
                        d_weight[c][z][y][x] <= w_cnn_weight[(c*`ST2_Conv_CI*`KY*`KX + z*`KY*`KX + y*`KX + x)*`W_BW +: `W_BW];
    end


genvar ci_inst;
generate
	for(ci_inst = 0; ci_inst < `ST2_Conv_CO; ci_inst = ci_inst + 1) begin : gen_ci_inst
        
		assign	w_in_valid[ci_inst] = r_w_valid  [V_LATENCY-1 : 0] ; 
		stage2_cnn_acc_ci u_stage2_cnn_acc_ci(
	    .clk             (clk         ),
	    .reset_n         (reset_n     ),
	    .i_cnn_weight    (w_cnn_weight[ci_inst*`ST2_Conv_CI*`KX*`KY*`W_BW +: `ST2_Conv_CI*`KX*`KY*`W_BW]),
	    .i_in_valid      (w_in_valid[ci_inst]),
	    .i_in_fmap       (window),
	    .o_ot_valid      (w_ot_valid[ci_inst]),
	    .o_ot_ci_acc     (w_ot_ci_acc[ci_inst*(`ACI_BW) +: (`ACI_BW)])         
	    );
	end
endgenerate



//debug 
reg signed [`ACI_BW-1:0] d_ot_ci_acc [0:`ST2_Conv_CO-1];
genvar ch;
generate
        for (ch= 0; ch<`ST2_Conv_CO ; ch= ch+1) begin
            always @(posedge clk or negedge reset_n) begin
                if(!reset_n) begin
                    d_ot_ci_acc[ch] <= 0;
                end else if(&w_ot_valid)begin
                    d_ot_ci_acc[ch] <= $signed(w_ot_ci_acc[ch*(`ACI_BW) +: (`ACI_BW)]);
                end
            end
        end
endgenerate

//==============================================================================
// add_bias = acc + bias
//==============================================================================

wire   signed   [`ST2_Conv_CO*`AB_BW-1 : 0]   add_bias  ;
reg    signed   [`ST2_Conv_CO*`AB_BW-1 : 0]   r_add_bias;

wire   signed   [`ST2_Conv_CO*`B_BW-1  : 0]   w_cnn_bias;
assign  w_cnn_bias = i_cnn_bias;

//debug
reg signed [`AB_BW-1:0] d_r_add_bias [0:`ST2_Conv_CO-1];

genvar  add_idx;
generate
    for (add_idx = 0; add_idx < `ST2_Conv_CO; add_idx = add_idx + 1) begin : gen_add_bias
        assign  add_bias[add_idx*`AB_BW +: `AB_BW] = $signed(w_ot_ci_acc[add_idx*(`ACI_BW) +: `ACI_BW]) + $signed(w_cnn_bias[add_idx*`B_BW +: `B_BW]);

        always @(posedge clk or negedge reset_n) begin
            if(!reset_n) begin
                r_add_bias[add_idx*`AB_BW +: `AB_BW]   <= 0;
                d_r_add_bias[add_idx] <= 0;
            end else if(&w_ot_valid) begin
                r_add_bias[add_idx*`AB_BW +: `AB_BW]   <= $signed(add_bias[add_idx*`AB_BW +: `AB_BW]);
                d_r_add_bias[add_idx] <= $signed(add_bias[add_idx*`AB_BW +: `AB_BW]);
       
            end
        end
    end
endgenerate




//==============================================================================
// Activation
//==============================================================================
// bias까지 더하고 나서 output channel 3개에 대한 1point (1point에 대해서 bit width는 = `O_F_BW(=33))
// 3ch * 
reg [`ST2_Conv_CO * `O_F_BW-1:0] act_relu;

//debug
reg [`O_F_BW-1:0] d_act_relu   [0:`ST2_Conv_CO-1];
reg [`O_F_BW-2:0] d_r_act_relu [0:`ST2_Conv_CO-1];



// 3ch * 
reg [`ST2_Conv_CO * (`O_F_BW-1)-1:0] r_act_relu;

	    always @ (*) begin
            for (i = 0; i < `ST2_Conv_CO; i = i + 1) begin
                if (r_add_bias[i*`O_F_BW +: `O_F_BW] >>> (`O_F_BW-1)) begin// MSB가 1이면 음수
                    act_relu[i*`O_F_BW +: `O_F_BW] = 0;
                    //debug
                    d_act_relu [i] = 0;
                end else begin
                    act_relu[i*`O_F_BW +: `O_F_BW] = $signed(r_add_bias[i*`O_F_BW +: `O_F_BW]);
                    d_act_relu [i] =  $signed(r_add_bias[i*`O_F_BW +: `O_F_BW]);
                end
            end
	    end

        always @(posedge clk or negedge reset_n) begin
            if(!reset_n) begin
                r_act_relu <= 0;
                for (i = 0; i < `ST2_Conv_CO; i = i + 1) begin
                    d_r_act_relu[i] <= 0;
                end                
            end else if(r_valid[LATENCY-2]) begin
                for (i = 0; i < `ST2_Conv_CO; i = i + 1) begin
                    r_act_relu[i*(`O_F_BW-1) +: `O_F_BW-1] <= $signed(act_relu[i*`O_F_BW +: `O_F_BW-1]); // 하위 32비트만 저장
                    d_r_act_relu[i] <= $signed(act_relu[i*`O_F_BW +: `O_F_BW-1]);
                end
            end
        end



assign o_ot_valid = r_valid[LATENCY-1];
assign o_ot_fmap  = r_act_relu;

endmodule

