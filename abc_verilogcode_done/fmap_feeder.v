`timescale 1ns / 1ps
`include "defines_cnn_core.v"

module fmap_feeder(
    input clk,
    input reset_n,
    input i_valid,                      // 1클럭만 주면 내부에서 자동 시작,
    input [31:0] i_pixel,
    input [3:0] sw,
    output [`ISP_BW-1:0] o_pixel,    // cnn_top의 i_pixel에 연결
    output o_out_valid
);

    // reg [$clog2(`TOTAL_PIXELS)-1:0] addr;

    // reg [`ISP_BW-1:0] pixel_reg;
    // reg valid_reg;
    // reg is_sending;
    // reg is_done;


    reg [`ISP_BW-1:0] selected_pixel;

    reg [1:0] state, state_next;
    reg [`ISP_BW-1:0] pixel_reg, pixel_next;
    reg [$clog2(`TOTAL_PIXELS)-1:0] addr_reg, addr_next;
    reg valid_reg, valid_next;


    localparam IDLE = 0, SENDING = 1, DONE = 2;

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            state <= IDLE;
            addr_reg <= 0;
            pixel_reg <= 0;
            valid_reg <= 0;
        end else begin
            state <= state_next;
            addr_reg <= addr_next;
            pixel_reg <= pixel_next;
            valid_reg <= valid_next;
        end
    end


    always @(*) begin
        state_next = state;
        addr_next = addr_reg;
        pixel_next = pixel_reg;
        valid_next = valid_reg;
        case (state)
            IDLE    : begin
                if(i_valid) begin
                    state_next = SENDING;
                end
            end
            SENDING : begin
                    if (addr_reg < `TOTAL_PIXELS) begin
                        pixel_next = selected_pixel;
                        valid_next = 1;
                        addr_next = addr_reg + 1;  
                    end else begin
                        pixel_next = 0;
                        valid_next = 0;
                        addr_next = 0;
                        state_next <= DONE;
                    end           
            end
            DONE    : begin
                state_next = IDLE;
            end
        endcase
    end

    assign o_pixel = pixel_reg;
    assign o_out_valid = valid_reg;    

endmodule