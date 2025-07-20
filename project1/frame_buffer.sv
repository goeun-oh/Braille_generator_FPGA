`timescale 1ns / 1ps

module frame_buffer (
    // write side
    input  logic        wclk,
    input  logic        we,
    input  logic [16:0] wAddr,  // 320 * 240 = 76800, 17비트 주소
    input  logic [15:0] wData,
    // read side
    input  logic        rclk,
    input  logic        oe,     // output enable
    input  logic [16:0] rAddr,
    output logic [15:0] rData
);
    logic [15:0] mem[0:(320*240 - 1)];

    // write side
    always_ff @(posedge wclk) begin : write_side
        if (we) begin
            mem[wAddr] <= wData;
        end
    end

    // read side
    always_ff @(posedge rclk) begin : read_side
        if (oe) begin
            rData <= mem[rAddr];
        end
    end

endmodule
