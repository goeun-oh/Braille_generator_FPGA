`timescale 1ns / 1ps

module fmap_feeder #(
    parameter I_F_BW = 8,
    parameter IX = 28,
    parameter IY = 28,
    parameter TOTAL_PIXELS = IX * IY
)(
    input clk,
    input reset_n,
    input i_valid,                      // 1클럭만 주면 내부에서 자동 시작
    output [I_F_BW-1:0] o_pixel,    // cnn_top의 i_pixel에 연결
    output o_out_valid
);

    reg [I_F_BW-1:0] fmap_rom [0:TOTAL_PIXELS-1];
    reg [$clog2(TOTAL_PIXELS)-1:0] addr;

    reg [I_F_BW-1:0] pixel_reg;
    reg valid_reg;
    reg is_sending;
    reg is_done;

    assign o_pixel = pixel_reg;
    assign o_out_valid = valid_reg;

    initial begin
        $readmemh("input_fmap.mem", fmap_rom);
    end

    always @(*) begin
        if(i_valid) begin
            is_sending <=1;
        end else if (is_done) begin
            is_sending <=0;
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            addr <= 0;
            pixel_reg <= 0;
            valid_reg <= 0;
            is_done <=0;
        end else begin
            if (is_sending) begin
                pixel_reg <=0;
                if (addr < TOTAL_PIXELS) begin
                    pixel_reg <= fmap_rom[addr];
                    valid_reg <= 1;
                    addr <= addr + 1;
                    is_done <=0;
                end else if (addr == TOTAL_PIXELS)begin
                    valid_reg <= 0;
                    pixel_reg <= 0;
                    addr <=0;
                    is_done <= 1;
                end
            end else begin
                valid_reg <= 0;
                addr <=0;
                pixel_reg <=0;
                is_done <=0;
            end
        end
    end

endmodule
