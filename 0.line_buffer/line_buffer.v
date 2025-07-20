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
    output [$clog2(IX)-1:0] o_if_x,
    output [$clog2(IY)-1:0] o_if_y
);

    reg r_valid;
    reg [$clog2(IX)-1:0] x_cnt;
    reg [$clog2(IY)-1:0] y_cnt;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_cnt <=0;
            y_cnt <=0;
        end else if (i_in_valid) begin
            if (x_cnt == IX -1) begin
                x_cnt <=0;
                y_cnt <= y_cnt + 1;
            end else begin
                if (y_cnt == IY-1) begin
                    y_cnt <=0;
                end
                if (x_cnt >= KX -1 && y_cnt >= KY-1) begin
                    r_valid <=1;
                end else begin
                    r_valid <= 0;
                end
                x_cnt <= x_cnt + 1;
            end
        end
    end

    reg [I_F_BW-1:0] line_buf[0:KY-1][0:IX-1];  // 4줄만 저장. 최신 줄은 현재 pixel로 채움
    integer i;
    always @(posedge clk) begin
        if (i_in_valid) begin
            for (i=0; i < KY -1; i = i+1) begin
                line_buf[i][x_cnt] <= line_buf[i+1][x_cnt];
            end
            line_buf[KY-1][x_cnt] <= i_in_pixel;
        end
    end

    reg [KX*KY*I_F_BW-1:0] r_window;   

    integer wy, wx;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_window <= 0;
        end else if (r_valid && (x_cnt >= KX-1) && (y_cnt >= KY-1)) begin
            for (wy =0; wy <KY; wy = wy + 1) begin
                for (wx =0; wx <KX; wx = wx +1) begin
                    r_window[(wy*KX + wx)*I_F_BW +: I_F_BW] <=line_buf[wy][x_cnt -1 - (KX - 1 - wx)];
                end
            end
        end
    end
    
    reg r_window_valid;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_window_valid <=0;
        end else begin
            r_window_valid <= (x_cnt >= KX-1) && (y_cnt >= KY-1);
        end
    end

    assign o_window = r_window;
    assign o_window_valid = r_window_valid;

    reg [$clog2(IX)-1:0] r_if_x;
    reg [$clog2(IY)-1:0] r_if_y;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_if_x <= 0;
            r_if_y <= 0;
        end else if (r_valid && (x_cnt >= KX - 1) && (y_cnt >= KY - 1)) begin
            r_if_x <= x_cnt - (KX - 1);
            r_if_y <= y_cnt - (KY - 1);
        end
    end

    assign o_if_x = r_if_x;
    assign o_if_y = r_if_y;

endmodule