
`timescale 1ns / 1ps

`include "defines_cnn_core.v"

module stage2_pooling(
input                                                           clk            ,
input                                                           reset_n        ,
input                                                           i_in_valid     ,
input     [`ST2_Pool_IBW -1 : 0]                                 i_in_fmap       ,//1point(19bit)
output                                                          o_ot_valid     ,
output    [`ST2_Pool_IBW -1 : 0]                                o_ot_fmap        //1point(19bit)
    );

    // localparam LATENCY = 2;
    localparam COL = `ST2_Pool_X; //24
    localparam ROW = `ST2_Pool_Y; //24

    integer i;

//==============================================================================
// define max pooling function
//==============================================================================
    // //2x2 window
    // function [`ST2_Pool_IBW:0] max_pixel;
    //     input [2*2*`ST2_Pool_IBW-1 : 0] fmap; // 2x2x(19bit) window
    //     reg   [`ST2_Pool_IBW-1:0] a, b, c, d;
    //     reg   [`ST2_Pool_IBW-1:0] max1, max2, max_pool;

    //     begin
    //         a = fmap[0               +: `ST2_Pool_IBW];
    //         b = fmap[1*`ST2_Pool_IBW +: `ST2_Pool_IBW];
    //         c = fmap[2*`ST2_Pool_IBW +: `ST2_Pool_IBW];
    //         d = fmap[3*`ST2_Pool_IBW +: `ST2_Pool_IBW];
    //         max1 = (a > b) ? a : b;
    //         max2 = (c > d) ? c : d;
    //         max_pool = (max1 > max2) ? max1 : max2;
    //         max_pixel = max_pool;            
    //     end
    // endfunction

    
//==============================================================================
// row,col_counter
//==============================================================================
    reg [$clog2(ROW)-1:0] row;
    reg [$clog2(COL)-1:0] col;
    reg [$clog2(ROW)-1:0] row_delay;
    reg [$clog2(COL)-1:0] col_delay;

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

    // 1clk_delay row,col signal
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            row_delay <= 0;
            col_delay <= 0;  

        end else begin
            row_delay <= row;
            col_delay <= col;

        end 
    end    


//==============================================================================
// Line Buffer
//==============================================================================
    // 19bit line_buufer0 [0:23]
    reg [`ST2_Pool_IBW-1:0] line_buffer0 [0:`ST2_Pool_X-1];    
    reg [`ST2_Pool_IBW-1:0] line_buffer1 [0:`ST2_Pool_X-1];

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            for (i = 0; i < `ST2_Pool_X; i = i + 1) begin
                line_buffer0[i] <= 0;
                line_buffer1[i] <= 0;
            end
        end else begin
            if((i_in_valid)) begin 
                line_buffer1[col] <= i_in_fmap; // receive 1px data to Line Buffer        
                if(!col) begin
                    for (i = 0; i < `ST2_Pool_X; i = i + 1) begin
                        line_buffer0[i] <= line_buffer1[i];
                    end
                end
            end
        end
    end    


//==============================================================================
// apply max pooling function
//==============================================================================
   

// reg signed [`ST2_Pool_IBW:0] max_pixel;

    reg signed[`ST2_Pool_IBW-1:0] max_buffer_0;
    reg signed[`ST2_Pool_IBW-1:0] max_buffer_1;
    reg signed[`ST2_Pool_IBW-1:0] max_data;
    reg r_valid;
    reg r_valid2;

    // always @(*) begin
    //     o_pooling = max_pixel({
    //         line_buffer0[col_delay-1], line_buffer0[col_delay],
    //         line_buffer1[col_delay-1], line_buffer1[col_delay]
    //         });        
    // end

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            max_buffer_0 <= 0;
            max_buffer_1 <= 0;
            r_valid      <= 0;
        end else if( (row_delay[0]) && (col_delay[0]) ) begin
            max_buffer_0 <= (line_buffer0[col_delay-1] > line_buffer1[col_delay-1]) ? line_buffer0[col_delay-1] : line_buffer1[col_delay-1];
            max_buffer_1 <= (line_buffer0[col_delay]   > line_buffer1[col_delay])   ? line_buffer0[col_delay] : line_buffer1[col_delay];
            r_valid     <= 1;
        end else begin
            max_buffer_0 <= 0;
            max_buffer_1 <= 0;            
            r_valid      <= 0;
        end
    end    


    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            max_data       <= 0;
            r_valid2       <= 0;
        end else if( r_valid ) begin
            max_data       <= (max_buffer_0 > max_buffer_1) ? max_buffer_0 : max_buffer_1;
            r_valid2       <= r_valid;
        end else begin
            max_data       <= 0;
            r_valid2       <= 0;
        end
    end    


assign o_ot_fmap = max_data;
assign o_ot_valid =  r_valid2;


endmodule

