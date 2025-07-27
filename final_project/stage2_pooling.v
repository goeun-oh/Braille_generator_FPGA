`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/19 18:12:06
// Design Name: 
// Module Name: stage2_pooling
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

`include "timescale.v"
`include "defines_cnn_core.v"

module stage2_pooling(
input                                                           clk         	,
input                                                           reset_n     	,
input                                                           i_in_valid  	,
input     [`ST2_Pool_IBW -1 : 0]  	                            i_in_fmap    	,//1point(19bit)
output    reg                                                   o_ot_valid  	,
output    [`ST2_Pool_IBW -1 : 0]                                o_ot_fmap        //1point(19bit)
    );

    localparam COL = `ST2_Pool_X; //24
    localparam ROW = `ST2_Pool_Y; //24
    
//==============================================================================
// define max pooling function
//==============================================================================
    //2x2 window
    function [`ST2_Pool_IBW:0] max_pixel;
        input [2*2*`ST2_Pool_IBW-1 : 0] fmap; // 2x2x(19bit) window
        reg   [`ST2_Pool_IBW-1:0] a, b, c, d;
        reg   [`ST2_Pool_IBW-1:0] max1, max2, max_pool;

        begin
            a = fmap[0               +: `ST2_Pool_IBW];
            b = fmap[1*`ST2_Pool_IBW +: `ST2_Pool_IBW];
            c = fmap[2*`ST2_Pool_IBW +: `ST2_Pool_IBW];
            d = fmap[3*`ST2_Pool_IBW +: `ST2_Pool_IBW];
            max1 = (a > b) ? a : b;
            max2 = (c > d) ? c : d;
            max_pool = (max1 > max2) ? max1 : max2;
            max_pixel = max_pool;            
        end
    endfunction
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
// Line Buffer
//==============================================================================
    //1*24*(19bit)
    // reg [`ST2_Pool_X * `ST2_Pool_IBW-1:0] line_buffer0; //-> 가장 최근 데이터 저장, 한줄 다 채우고 line_buffer1로 shift
    //1*24*(19bit)
    // reg [`ST2_Pool_X * `ST2_Pool_IBW-1:0] line_buffer1;


    // 19bit line_buufer0 [0:23]
    reg [`ST2_Pool_IBW-1:0] line_buffer0 [0:`ST2_Pool_X-1];
    reg [`ST2_Pool_IBW-1:0] line_buffer1 [0:`ST2_Pool_X-1];    


    integer i;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            // line_buffer0   <= 0;
            // line_buffer1   <= 0;
            for (i = 0; i < `ST2_Pool_X; i = i + 1) begin
                line_buffer0[i] <= 0;
                line_buffer1[i] <= 0;
            end
        end else begin
            if((i_in_valid) && (!col)) begin // c가 0되면 line_buffer 1로 shift
                for (i = 0; i < `ST2_Pool_X; i = i + 1) begin
                    line_buffer1[i] <= line_buffer0[i];
                end
            end
        end
    end    

//==============================================================================
// receive 1px data to Line Buffer
//==============================================================================
    always @(posedge clk) begin
        if (i_in_valid) begin
            // line_buffer0[col*`ST2_Pool_IBW+:`ST2_Pool_IBW] <= i_in_fmap;
            // valid신호가 들어올 때만 data를 받아옴 
            line_buffer0[col] <= i_in_fmap;
        end
    end


//==============================================================================
// apply max pooling function
//==============================================================================
   
    reg [`ST2_Pool_IBW-1:0] o_pooling;
    reg [`ST2_Pool_IBW-1:0] r_o_pooling;

    always @(*) begin
        // o_pooling = max_pixel({line_buffer1[col*`ST2_Pool_IBW+:`ST2_Pool_IBW],line_buffer1[(col+1)*`ST2_Pool_IBW+:`ST2_Pool_IBW],    line_buffer0[col*`ST2_Pool_IBW+:`ST2_Pool_IBW],line_buffer0[(col+1)*`ST2_Pool_IBW+:`ST2_Pool_IBW]});        
        o_pooling = max_pixel({
            line_buffer1[col-2],
            line_buffer1[col-1],
            line_buffer0[col-2],
            line_buffer0[col-1]});        
    end

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            o_ot_valid <= 0;
            r_o_pooling   <= 0;
        end else if( !frame_flag ) begin
            if((row[0]) && (!col[0]) && !col_flag) begin
                o_ot_valid <= 1;
                r_o_pooling <= o_pooling;
            end else if (!row[0] && col_flag)begin
                o_ot_valid <= 1;
                r_o_pooling <= max_pixel({line_buffer1[`ST2_Pool_X-2], line_buffer1[`ST2_Pool_X-1], line_buffer0[`ST2_Pool_X-2], line_buffer0[`ST2_Pool_X-1]});      
            end else begin
                o_ot_valid <= 0;
            end
        end else if (frame_flag && col_flag) begin
            o_ot_valid <= 1;
            r_o_pooling <= max_pixel({line_buffer1[`ST2_Pool_X-2], line_buffer1[`ST2_Pool_X-1], line_buffer0[`ST2_Pool_X-2], line_buffer0[`ST2_Pool_X-1]});      
        end else begin
            o_ot_valid <= 0;
        end
    end
assign o_ot_fmap = r_o_pooling;



endmodule

