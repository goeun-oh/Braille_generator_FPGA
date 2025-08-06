`include "defines_cnn_core.v"

module alpha_decoder (
    input clk,
    input reset_n,
    input                         i_valid,
    input  [$clog2(`core_CO)-1:0] index_info,
    output reg [                 7:0] o_alpha,
    output reg                        o_valid,
    output [3:0] led,
    output led_r,
    output led_g,
    output led_b
);

    reg [7:0] alpha;
    //assign o_alpha = alpha;
    //assign o_valid = i_valid;


    reg reg_led_r, reg_led_g, reg_led_b;
    reg [3:0] reg_led;
    reg [3:0] next_led;
    reg next_led_r, next_led_g, next_led_b;

    assign led_r = reg_led_r;
    assign led_g = reg_led_g;
    assign led_b = reg_led_b;
    assign led = reg_led;

    always @(posedge clk, negedge reset_n) begin
        if (!reset_n) begin
            o_alpha <= 0;
            o_valid <= 0;
        end else begin
            o_valid <= i_valid;
            if (i_valid) begin
                o_alpha <= alpha;
            end
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            reg_led_r <=0;
            reg_led_g <=0;
            reg_led_b <=0;
            reg_led <=0;
        end else if (i_valid)begin
            reg_led_r <=next_led_r;
            reg_led_g <=next_led_g;
            reg_led_b <=next_led_b;            
            reg_led <= next_led;
        end
    end

    always @(*) begin
        next_led_r = reg_led_r;
        next_led_g = reg_led_g;
        next_led_b = reg_led_b;
        next_led = reg_led;
        case (index_info)
        //a
            5'd0: begin
                alpha = 8'h61;
                next_led = 4'd0;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //b
            5'd1: begin
                alpha = 8'h62;
                next_led = 4'd1;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //c
            5'd2: begin
                alpha = 8'h63;
                next_led = 4'd2;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //d
            5'd3: begin
                alpha = 8'h64;
                next_led = 4'd3;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //e
            5'd4: begin
                alpha = 8'h65;
                next_led = 4'd4;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //f
            5'd5: begin
                alpha = 8'h66;
                next_led = 4'd5;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //g
            5'd6: begin
                alpha = 8'h67;
                next_led = 4'd6;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //h
            5'd7: begin
                alpha = 8'h68;
                next_led = 4'd7;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //i
            5'd8: begin
                alpha = 8'h69;
                next_led = 4'd8;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //j
            5'd9: begin
                alpha = 8'h6A;
                next_led = 4'd9;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //k
            5'd10: begin
                alpha = 8'h6B;
                next_led = 4'd10;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //l
            5'd11: begin
                alpha = 8'h6C;
                next_led = 4'd11;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //m
            5'd12: begin
                alpha = 8'h6D;
                next_led = 4'd12;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
            //n
            5'd13: begin
                alpha = 8'h6E;
                next_led = 4'd13;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end

            //o
            5'd14: begin
                alpha = 8'h6F;
                next_led = 4'd14;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end

            //p
            5'd15: begin
                alpha = 8'h70;
                next_led = 4'd15;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end

            //q
            5'd16: begin
                alpha = 8'h71;
                next_led = 4'd0;
                next_led_r = 0;
                next_led_g = 0;
                next_led_b = 1;
            end

            //r
            5'd17: begin
                alpha = 8'h72;
                next_led = 4'd1;
                next_led_r = 0;
                next_led_g = 0;
                next_led_b = 1;
            end

            //s
            5'd18: begin
                alpha = 8'h73;
                next_led = 4'd2;
                next_led_r = 0;
                next_led_g = 0;
                next_led_b = 1;
            end

            //t
            5'd19: begin
                alpha = 8'h74;
                next_led = 4'd3;
                next_led_r = 0;
                next_led_g = 0;
                next_led_b = 1;
            end
            5'd20: begin
                alpha = 8'h75;
                next_led = 4'd4;
                next_led_r = 0;
                next_led_g = 0;
                next_led_b = 1;
            end
            5'd21: begin
                alpha = 8'h76;
                next_led = 4'd5;
                next_led_r = 0;
                next_led_g = 0;
                next_led_b = 1;
            end
            5'd22: begin
                alpha = 8'h77;
                next_led = 4'd6;
                next_led_r = 0;
                next_led_g = 0;
                next_led_b = 1;
            end
            5'd23: begin
                alpha = 8'h78;
                next_led = 4'd7;
                next_led_r = 0;
                next_led_g = 0;
                next_led_b = 1;
            end
            5'd24: begin
                alpha = 8'h79;
                next_led = 4'd8;
                next_led_r = 0;
                next_led_g = 0;
                next_led_b = 1;
            end
            5'd25: begin
                alpha = 8'h7A;
                next_led = 4'd9;
                next_led_r = 0;
                next_led_g = 0;
                next_led_b = 1;
            end
            default: begin
                alpha = 8'h61;
                next_led = 4'd0;
                next_led_r = 1;
                next_led_g = 1;
                next_led_b = 1;
            end
        endcase
        //case (index_info)
        //    2'd0: begin
        //        alpha = 8'h61;
        //    end
        //    2'd1: begin
        //        alpha = 8'h62;
        //    end
        //    2'd2: begin
        //        alpha = 8'h63;
        //    end
        //endcase
    end

endmodule
