`timescale 1ns / 1ps
`include "defines_cnn_core.v"

module fmap_feeder(
    input clk,
    input reset_n,
    input i_valid,                      // 1클럭만 주면 내부에서 자동 시작,
    input [3:0] sw,
    output [`ISP_BW-1:0] o_pixel,    // cnn_top의 i_pixel에 연결
    output o_out_valid
);

    reg [`ISP_BW-1:0] fmap_rom_a_0 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_a_1 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_a_2 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_a_3 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_b_0 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_b_1 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_b_2 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_b_3 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_c_0 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_c_1 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_c_2 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_c_3 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_d_0 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_d_1 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_d_2 [0:`TOTAL_PIXELS-1];
    reg [`ISP_BW-1:0] fmap_rom_d_3 [0:`TOTAL_PIXELS-1];

    // reg [$clog2(`TOTAL_PIXELS)-1:0] addr;

    // reg [`ISP_BW-1:0] pixel_reg;
    // reg valid_reg;
    // reg is_sending;
    // reg is_done;



    initial begin
        $readmemh("a_1_rgb.mem", fmap_rom_a_0);
        $readmemh("b_1_rgb.mem", fmap_rom_a_1);
        $readmemh("c_1_rgb.mem", fmap_rom_a_2);
        $readmemh("d_1_rgb.mem", fmap_rom_a_3);

        $readmemh("e_1_rgb.mem", fmap_rom_b_0);
        $readmemh("f_1_rgb.mem", fmap_rom_b_1);
        $readmemh("g_1_rgb.mem", fmap_rom_b_2);
        $readmemh("h_1_rgb.mem", fmap_rom_b_3);

        $readmemh("i_1_rgb.mem", fmap_rom_c_0);
        $readmemh("j_1_rgb.mem", fmap_rom_c_1);
        $readmemh("k_1_rgb.mem", fmap_rom_c_2);
        $readmemh("l_1_rgb.mem", fmap_rom_c_3);

        $readmemh("m_1_rgb.mem", fmap_rom_d_0);
        $readmemh("n_1_rgb.mem", fmap_rom_d_1);
        $readmemh("o_1_rgb.mem", fmap_rom_d_2);
        $readmemh("p_1_rgb.mem", fmap_rom_d_3);
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
    //             if (addr < `TOTAL_PIXELS) begin
    //                 pixel_reg <= selected_pixel;
    //                 valid_reg <= 1;
    //                 addr <= addr + 1;
    //                 is_done <=0;
    //             end else if (addr == `TOTAL_PIXELS)begin
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
    reg [`ISP_BW-1:0] selected_pixel;

    reg [1:0] state, state_next;
    reg [`ISP_BW-1:0] pixel_reg, pixel_next;
    reg [$clog2(`TOTAL_PIXELS)-1:0] addr_reg, addr_next;
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
            
            4'd12: selected_pixel = fmap_rom_d_0[addr_reg];
            4'd13: selected_pixel = fmap_rom_d_1[addr_reg];
            4'd14: selected_pixel = fmap_rom_d_2[addr_reg];
            4'd15: selected_pixel = fmap_rom_d_3[addr_reg];

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