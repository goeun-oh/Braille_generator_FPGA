`timescale 1ns/1ps
module line_buffer #(
    parameter I_F_BW =8,
    parameter IX = 28,
    parameter IY = 28,
    parameter KX = 5,
    parameter KY = 5
)(
    input clk,
    input reset_n,

    input i_in_valid,
    input [I_F_BW-1:0] i_in_pixel,
    
    output o_window_valid,
    output [KX*KY*I_F_BW-1:0] o_window,
    output [I_F_BW*KX-1:0] o_line_buf
);

    parameter LATENCY = 2;
    reg [$clog2(IX)-1:0] x_cnt;
    reg [$clog2(IY)-1:0] y_cnt;
    reg [KX*KY*I_F_BW-1:0] r_window_d1;

    //디버깅
    reg [I_F_BW-1:0] r_line_buf;
    
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_cnt <=0;
            y_cnt <=0;
        end else if (i_in_valid) begin
            if (x_cnt == IX-1) begin
                x_cnt <=0;
                if (y_cnt == IY-1) begin
                    y_cnt <=0;
                end else begin
                    y_cnt <= y_cnt + 1;
                end
            end else begin
               x_cnt <= x_cnt + 1;
            end
        end
    end

    reg [I_F_BW-1:0] line_buf[0:KY-1][0:IX-1];  // 4줄만 저장. 최신 줄은 현재 pixel로 채움
    
    integer i;
    always @(posedge clk) begin
        for (i=0; i < KY -1; i = i+1) begin
            line_buf[i][x_cnt] <= line_buf[i+1][x_cnt];
        end
        line_buf[KY-1][x_cnt] <= i_in_pixel;
    end

    reg [KX*KY*I_F_BW-1:0] r_window;   

    //디버깅
    assign o_line_buf = r_line_buf;
    wire r_window_valid;
    
    reg [LATENCY-1:0] shift_window_valid;
    assign r_window_valid = (x_cnt >= KX) && (y_cnt >= KY-1);
    
    integer wy, wx;
    always @(*) begin
        if (r_window_valid) begin
            for (wy =0; wy <KY; wy = wy + 1) begin
                for (wx =0; wx <=KX; wx = wx +1) begin
                    r_window[(wy*KX + wx)*I_F_BW +: I_F_BW] =line_buf[wy][x_cnt-(KX- wx)];
                    //r_line_buf[wx *I_F_BW +: I_F_BW] <= line_buf[wy][x_cnt-1-(KX-1-wx)];       
                end
            end
        end
    end


     reg  [LATENCY-1 : 0] r_wait_valid;
     always @(posedge clk or negedge reset_n) begin
         if (!reset_n) begin
             r_wait_valid <= {LATENCY{1'b0}};
         end else begin
             r_wait_valid[LATENCY-2] <= i_in_valid;
             r_wait_valid[LATENCY-1] <= r_wait_valid[LATENCY-2];
         end
     end

    assign o_window = r_window;
    assign o_window_valid = r_window_valid;
    //assign o_window_valid = r_window_valid;


endmodule