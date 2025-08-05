`timescale 1ns/1ps
`include "defines_cnn_core.v"
module line_buffer (
    input clk,
    input reset_n,

    input i_in_valid,
    input [`ST1_I_F_BW-1:0] i_in_pixel,
    
    output o_window_valid,
    output [`KX*`KY*`ST1_I_F_BW-1:0] o_window
);

    parameter LATENCY = 2;
    reg [$clog2(`ST1_IX)-1:0] x_cnt;
    reg [$clog2(`ST1_IY)-1:0] y_cnt;
    integer i;
    integer j;
    //디버깅
    reg [`ST1_I_F_BW-1:0] r_line_buf;
    reg flag;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_cnt <=0;
            y_cnt <=0;
            flag <=0;
        end else if (i_in_valid) begin
            if (x_cnt == `ST1_IX-1) begin
                x_cnt <=0;
                flag <=1;
                if (y_cnt == `ST1_IY-1) begin
                    y_cnt <= 0;
                end else begin
                    y_cnt <= y_cnt +1;
                end
            end else begin
                flag <=0;
                x_cnt <= x_cnt + 1;
            end
        end 
    end

    reg [`ST1_I_F_BW-1:0] line_buf[0:`KY-1][0:`ST1_IX-1];  // 4줄만 저장. 최신 줄은 현재 pixel로 채움
    

    always @(posedge clk) begin
        if(i_in_valid) begin
            for (i=0; i < `KY-1; i = i+1) begin
                line_buf[i][x_cnt] <= line_buf[i+1][x_cnt];
            end
            line_buf[`KY-1][x_cnt] <= i_in_pixel;
        end
    end

    reg [`KX*`KY*`ST1_I_F_BW-1:0] r_window;   

    //디버깅
    assign o_line_buf = r_line_buf;
    reg r_window_valid;
    
    reg [LATENCY-1:0] shift_window_valid;
    reg [$clog2(`ST1_IX)-1:0] window_x_cnt;
    reg [$clog2(`ST1_IY)-1:0] window_y_cnt;

    integer wy, wx;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin            
            for (wy =0; wy <`KY; wy = wy + 1) begin
                for (wx =0; wx <`KX; wx = wx +1) begin
                    r_window[(wy*`KX + wx)*`ST1_I_F_BW +: `ST1_I_F_BW] =0;
                end
            end            
        end else begin
            for (wy =0; wy <`KY; wy = wy + 1) begin
                if(window_x_cnt >= `KX-1 ) begin
                    for (wx =0; wx <`KX; wx = wx +1) begin
                        r_window[(wy*`KX + wx)*`ST1_I_F_BW +: `ST1_I_F_BW] =line_buf[wy][window_x_cnt-(`KX-1- wx)];
                        //r_line_buf[wx *`ST1_I_F_BW +: `ST1_I_F_BW] <= line_buf[wy][x_cnt-1-(`KX-1-wx)];       
                    end
                end
            end            
        end
    end
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_window_valid <=0;
        end else begin
            if(x_cnt >= `KX -1 && y_cnt >= `KY-1) begin
                r_window_valid <= 1;
            end else if (window_x_cnt == `ST1_IX -1 && window_y_cnt == `ST1_IY -`KY) begin
                r_window_valid <=0;
            end
        end
    end


    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            window_x_cnt <=0;
            window_y_cnt <=0;
        end else begin
            if(r_window_valid) begin
                if(window_x_cnt == `ST1_IX-1) begin
                    window_x_cnt <= 0; 
                    if(window_y_cnt == `ST1_IY-`KY) begin
                        window_y_cnt <= 0;
                    end else begin
                        window_y_cnt <= window_y_cnt +1;
                    end
                end else begin
                    window_x_cnt <= window_x_cnt + 1;
                end
            end
        end
    end

    reg  [LATENCY-1 : 0] r_wait_valid;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_wait_valid <= {LATENCY{1'b0}};
        end else begin
            r_wait_valid[LATENCY-1] <= (window_x_cnt >= `KX-1);
            
        end
    end

    //debug
    integer k;
    reg [`ST1_I_F_BW-1:0] result_window [0:`KY-1][0:`KX-1];
    always @(posedge clk) begin
        if (r_wait_valid[LATENCY-1]) begin
            for (k= 0; k < `KY; k = k + 1) begin
                for (j= 0; j < `KX; j = j + 1) begin
                    result_window[k][j] <= o_window[(k*`KX+j)*`ST1_I_F_BW +: `ST1_I_F_BW];
                end
            end
        end
    end
    assign o_window = r_window;
    assign o_window_valid =  r_wait_valid[LATENCY-1] ;
    //assign o_window_valid = r_window_valid;


endmodule