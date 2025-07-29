`timescale 1ns / 1ps


module fmap_feeder #(
    parameter I_F_BW = 8,
    parameter IX = 28,
    parameter IY = 28,
    parameter TOTAL_PIXELS = IX * IY
)(
    input clk,
    input reset_n,
    input i_valid,                      // 1클럭만 주면 내부에서 자동 시작,
    input [3:0] sw,
    output [I_F_BW-1:0] o_pixel,    // cnn_top의 i_pixel에 연결
    output o_out_valid
);

    reg [I_F_BW-1:0] fmap_rom_a_0 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_a_1 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_a_2 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_a_3 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_b_0 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_b_1 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_b_2 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_b_3 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_c_0 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_c_1 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_c_2 [0:TOTAL_PIXELS-1];
    reg [I_F_BW-1:0] fmap_rom_c_3 [0:TOTAL_PIXELS-1];

    // reg [$clog2(TOTAL_PIXELS)-1:0] addr;

    // reg [I_F_BW-1:0] pixel_reg;
    // reg valid_reg;
    // reg is_sending;
    // reg is_done;



    initial begin
        $readmemh("a_1.mem", fmap_rom_a_0);
        $readmemh("a_1.mem", fmap_rom_a_1);
        $readmemh("a_1.mem", fmap_rom_a_2);
        $readmemh("a_1.mem", fmap_rom_a_3);
        $readmemh("a_1.mem", fmap_rom_b_0);
        $readmemh("a_1.mem", fmap_rom_b_1);
        $readmemh("a_1.mem", fmap_rom_b_2);
        $readmemh("a_1.mem", fmap_rom_b_3);
        $readmemh("a_1.mem", fmap_rom_c_0);
        $readmemh("a_1.mem", fmap_rom_c_1);
        $readmemh("a_1.mem", fmap_rom_c_2);
        $readmemh("a_1.mem", fmap_rom_c_3);
    end
    
    // always @(*) begin
    //     if (i_valid)
    //         is_sending <= 1;
    //     else if (is_done)
    //         is_sending <= 0;
    // end

    // always @(posedge clk or negedge reset_n) begin
    //     if (!reset_n) begin
    //         addr <= 0;
    //         pixel_reg <= 0;
    //         valid_reg <= 0;
    //         is_done <=0;
    //     end else begin
    //         if (is_sending) begin
    //             pixel_reg <=0;
    //             if (addr < TOTAL_PIXELS) begin
    //                 pixel_reg <= selected_pixel;
    //                 valid_reg <= 1;
    //                 addr <= addr + 1;
    //                 is_done <=0;
    //             end else if (addr == TOTAL_PIXELS)begin
    //                 valid_reg <= 0;
    //                 pixel_reg <= 0;
    //                 addr <=0;
    //                 is_done <= 1;
    //             end
    //         end else begin
    //             valid_reg <= 0;
    //             addr <=0;
    //             pixel_reg <=0;
    //             is_done <=0;
    //         end
    //     end
    // end
    reg [I_F_BW-1:0] selected_pixel;

    reg [1:0] state, state_next;
    reg [I_F_BW-1:0] pixel_reg, pixel_next;
    reg [$clog2(TOTAL_PIXELS)-1:0] addr_reg, addr_next;
    reg valid_reg, valid_next;

    always @(*) begin
        case(sw)
            4'd0: selected_pixel = fmap_rom_a_0[addr_reg];
            4'd1: selected_pixel = fmap_rom_a_1[addr_reg];
            4'd2: selected_pixel = fmap_rom_a_2[addr_reg];
            4'd3: selected_pixel = fmap_rom_a_3[addr_reg];
            4'd4: selected_pixel = fmap_rom_b_0[addr_reg];
            4'd5: selected_pixel = fmap_rom_b_1[addr_reg];
            4'd6: selected_pixel = fmap_rom_b_2[addr_reg];
            4'd7: selected_pixel = fmap_rom_b_3[addr_reg];
            4'd8: selected_pixel = fmap_rom_c_0[addr_reg];
            4'd9: selected_pixel = fmap_rom_c_1[addr_reg];
            4'd10: selected_pixel = fmap_rom_c_2[addr_reg];
            4'd11: selected_pixel = fmap_rom_c_3[addr_reg];
            default: selected_pixel = 8'd0;
        endcase
    end



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
                    if (addr_reg < TOTAL_PIXELS) begin
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