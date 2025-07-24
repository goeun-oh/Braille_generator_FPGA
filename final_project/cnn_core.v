/*******************************************************************************
Author: joohan.kim (https://blog.naver.`ST2_Conv_COm/chacagea)
Asso`ST2_Conv_CIated Filename: cnn_core.v
Purpose: verilog `ST2_Conv_COde to understand the CNN operation
License : https://github.`ST2_Conv_COm/matbi86/matbi_fpga_season_1/blob/main/LICENSE
Revision History: February 13, 2020 - initial release
*******************************************************************************/

`include "timescale.v"
`include "defines_cnn_core.v"

module cnn_core (
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
input                                                       clk         	;
input                                                       reset_n     	;
input                                                       i_in_valid  	;
input     signed [`ST2_Conv_CI * `ST2_Conv_IBW-1 : 0]  	    i_in_fmap    	;//3*(19bit) , 3ch에 대한 1point output
output                                                      o_ot_valid  	;
output    signed [`ST2_Conv_CO*(`O_F_BW-1)-1 : 0]  		    o_ot_fmap       ;    

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
            frame_flag <= 0;
            col_flag <= 0;     
        end else if(i_in_valid) begin
            if(col == COL-1) begin
                col <= 0;
                col_flag <= 1;
                if (row == ROW -1) begin
                    row <= 0 ;
                    frame_flag <= 1;
                end else begin
                    row <= row + 1;
                    frame_flag <= 0;
                end
            end else begin
                col <= col + 1;
                col_flag <= 0;
                frame_flag <= 0;
            end
        end
    end

//==============================================================================
// Line Buffer & 5x5 window register
//==============================================================================

    //(19bit)  3channel 5x24 line_buffer
    reg signed [`ST2_Conv_IBW-1:0] line_buffer [0:`ST2_Conv_CI-1][0:`KY-1][0:`ST2_Conv_X-1];

    //(19bit)  3channel 5x5  window
    reg signed [`ST2_Conv_CI * `KY*`KX * `ST2_Conv_IBW-1:0] window;

    integer i,j,k;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            for(k = 0; k<`ST2_Conv_CI ; k= k+1) begin
                for (j = 0; j < `KY; j=j+1) begin
                    for (i = 0; i < `ST2_Conv_X; i = i + 1) begin
                        line_buffer[k][j][i] <= 0;
                    end
                end
            end
        end else begin
            //col는 매 clk 0~11 증가
            //한 point씩 올리는 방식
            if(i_in_valid) begin // c가 0되면 line_buffer 1로 shift
                for (k = 0; k < `ST2_Conv_CI; k = k+1) begin
                    for (j = 0; j< `KY-1 ; j= j+1) begin
                        line_buffer[k][j+1][col] <= line_buffer[k][j][col];
                    end
                    // line_buffer[k][1][col] <= line_buffer[k][0][col];
                    // line_buffer[k][2][col] <= line_buffer[k][1][col];
                    // line_buffer[k][3][col] <= line_buffer[k][2][col];
                    // line_buffer[k][4][col] <= line_buffer[k][3][col];
                end
            end
        end
    end    

//==============================================================================
// receive 1px data to 3ch Line Buffer
//==============================================================================
    always @(posedge clk) begin
        if (i_in_valid) begin
            // valid신호가 들어올 때만 data를 받아옴,(col에 따라 위치가 다름)
            // 맨 첫번째 라인버퍼에만
            for (k = 0; k < `ST2_Conv_CI; k = k+1) begin  
                line_buffer[k][0][col] <= i_in_fmap[k*`ST2_Conv_IBW +: `ST2_Conv_IBW] ;
            end
        end
    end



    
//==============================================================================
// allocate data from line buffer to window, valid signal
//==============================================================================
//                   3 * 5 * 5 * (19bit)
// reg signed [`ST2_Conv_CI * `KY*`KX * `ST2_Conv_IBW-1:0] window;

reg window_valid;

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin

        end else if( row>=4 && col >=5 ) begin
            for (j = 0; j < `ST2_Conv_CI; j=j+1) begin
                //5*5*(19bit) 데이터 윈도우에 넣기
                window[j*`KY*`KX * `ST2_Conv_IBW +: `KY*`KX * `ST2_Conv_IBW] 
                    <= {line_buffer4[][j],
                        line_buffer3[][j],
                        line_buffer2[][j],
                        line_buffer1[][j],
                        line_buffer0[][j]};
            end

        end else if ( (row>=4 && !col[0]) | frame_flag ) begin
           
        end else begin
            o_ot_valid <= 0;
        end
    end








//==============================================================================
// Data Enable Signals 
//==============================================================================
wire    [LATENCY-1 : 0] 	ce;
reg     [LATENCY-1 : 0] 	r_valid;
wire    [`ST2_Conv_CO-1 : 0]          w_ot_valid;
always @(posedge clk or negedge reset_n) begin
    if(!reset_n) begin
        r_valid   <= {LATENCY{1'b0}};
    end else begin
        r_valid[LATENCY-2]  <= &w_ot_valid;
        r_valid[LATENCY-1]  <= r_valid[LATENCY-2];
    end
end

assign	ce = r_valid;

//==============================================================================
// acc instance
//==============================================================================

wire    [`ST2_Conv_CO-1 : 0]             w_in_valid;
wire    [`ST2_Conv_CO*(`ACI_BW)-1 : 0]  w_ot_ci_acc;
genvar ci_inst;
generate
	for(ci_inst = 0; ci_inst < `ST2_Conv_CO; ci_inst = ci_inst + 1) begin : gen_ci_inst
		wire    [`ST2_Conv_CO*`ST2_Conv_CI*  `KX*`KY  *`W_BW -1 : 0] w_cnn_weight; // 3x(3x5x5) x (7bit)
		wire    [`ST2_Conv_CI*`KX*`KY*`ST2_Conv_IBW-1 : 0]       w_in_fmap    	=  i_in_fmap[0 +: `ST2_Conv_CI*`KY*`KX*`ST2_Conv_IBW];
		assign	w_in_valid[ci_inst] = i_in_valid; 

        conv2_weight_rom #(
            .CHANNEL_ID(ci_inst)
        ) u_rom (
            //wire    [`ST2_Conv_CI*`KX*`KY*`W_BW-1 : 0]  	w_cnn_weight;에 ROM에 있는 CI개의 weight묶음을 넣어줌
            // Conv_CI = 3, KX =3, KY=3, W_BW=7

            .weight(w_cnn_weight[ci_inst*`ST2_Conv_CI*`KX*`KY*`W_BW +: `ST2_Conv_CI*`KX*`KY*`W_BW]) // 3x5x5

        );


		cnn_acc_ci u_cnn_acc_ci(
	    .clk             (clk         ),
	    .reset_n         (reset_n     ),
	    .i_cnn_weight    (w_cnn_weight[ci_inst*`ST2_Conv_CI*`KX*`KY*`W_BW +: `ST2_Conv_CI*`KX*`KY*`W_BW]),
	    .i_in_valid      (w_in_valid[ci_inst]),
	    .i_in_fmap       (w_in_fmap),
	    .o_ot_valid      (w_ot_valid[ci_inst]),
	    .o_ot_ci_acc     (w_ot_ci_acc[ci_inst*(`ACI_BW) +: (`ACI_BW)])         
	    );
	end
endgenerate

//==============================================================================
// add_bias = acc + bias
//==============================================================================

wire      [`ST2_Conv_CO*`AB_BW-1 : 0]   add_bias  ;
reg       [`ST2_Conv_CO*`AB_BW-1 : 0]   r_add_bias;
genvar  add_idx;
generate
    for (add_idx = 0; add_idx < `ST2_Conv_CO; add_idx = add_idx + 1) begin : gen_add_bias
            wire      [`ST2_Conv_CO*`B_BW-1  : 0]   w_cnn_bias;
            conv2_bias_rom #(
                .CHANNEL_ID(add_idx)
            ) u_rom (
                //wire    [`ST2_Conv_CI*`KX*`KY*`W_BW-1 : 0]  	w_cnn_weight;에 ROM에 있는 CI개의 weight묶음을 넣어줌
                .bias_out(w_cnn_bias[add_idx*`B_BW +: `B_BW])
            );
  
        assign  add_bias[add_idx*`AB_BW +: `AB_BW] = w_ot_ci_acc[add_idx*(`ACI_BW) +: `ACI_BW] + w_cnn_bias[add_idx*`B_BW +: `B_BW];

        always @(posedge clk or negedge reset_n) begin
            if(!reset_n) begin
                r_add_bias[add_idx*`AB_BW +: `AB_BW]   <= {`AB_BW{1'b0}};
            end else if(&w_ot_valid) begin
                r_add_bias[add_idx*`AB_BW +: `AB_BW]   <= add_bias[add_idx*`AB_BW +: `AB_BW];
            end
        end
    end
endgenerate

//==============================================================================
// Activation
//==============================================================================
// bias까지 더하고 나서 output channel 3개에 대한 1point (1point에 대해서 bit width는 = `O_F_BW(=33))
reg [`ST2_Conv_CO * `O_F_BW-1:0] act_relu;
reg [`ST2_Conv_CO * (`O_F_BW-1)-1:0] r_act_relu;

	    always @ (*) begin
            for (i = 0; i < `ST2_Conv_CO; i = i + 1) begin
                if (r_add_bias[i*`O_F_BW +: `O_F_BW] >>> (`O_F_BW-1)) // MSB가 1이면 음수
                    act_relu[i*`O_F_BW +: `O_F_BW] = {`O_F_BW{1'b0}};
                else
                    act_relu[i*`O_F_BW +: `O_F_BW] = r_add_bias[i*`O_F_BW +: `O_F_BW];
            end
	    end

        always @(posedge clk or negedge reset_n) begin
            if(!reset_n) begin
                r_act_relu <= 0;
            end else if(r_valid[LATENCY-2]) begin
                for (i = 0; i < `ST2_Conv_CO; i = i + 1)
                
                    r_act_relu[i*(`O_F_BW-1) +: `O_F_BW-1] <= act_relu[i*`O_F_BW +: `O_F_BW-1]; // 하위 32비트만 저장
            end
        end



assign o_ot_valid = r_valid[LATENCY-1];
assign o_ot_fmap  = r_act_relu;

endmodule

