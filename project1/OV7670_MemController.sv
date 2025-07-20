`timescale 1ns / 1ps

module OV7670_MemController (
    input  logic        pclk,
    input  logic        reset,
    input  logic        href,
    input  logic        v_sync,
    input  logic [ 7:0] ov7670_data,
    output logic        we,
    output logic [16:0] wAddr,
    output logic [15:0] wData
);
    logic [ 9:0] h_counter;  // 320 * 2 = 640(320 pixel)
    logic [ 7:0] v_counter;  // 240 line -> 2배로 늘어날 일은 없음
    logic [15:0] pix_data;

    // 두개를 합친거니까 나누기 2 해줘야함 -> 제일 하위 비트를 없애면 됨
    assign wAddr = v_counter * 320 + h_counter[9:1];
    assign wData = pix_data;

    always_ff @(posedge pclk, posedge reset) begin : h_sequence
        if (reset) begin
            pix_data  <= 0;
            h_counter <= 0;
            we        <= 1'b0;
        end else begin
            if (href == 1'b0) begin
                h_counter <= 0;
                we        <= 1'b0;
            end else begin
                h_counter <= h_counter + 1;
                if (h_counter[0] == 1'b0) begin  // 짝수 even data
                    pix_data[15:8] <= ov7670_data;  // 상위 8비트
                    we             <= 1'b0;
                end else begin  // 홀수 odd data
                    pix_data[7:0] <= ov7670_data;  // 하위 8비트
                    we            <= 1'b1;
                end
            end
        end
    end

    always_ff @(posedge pclk, posedge reset) begin : v_sequence
        if (reset) begin
            v_counter <= 0;
        end else begin
            if (v_sync) begin
                v_counter <= 0;
            end else begin
                if (h_counter == 640 - 1) begin
                    v_counter <= v_counter + 1;
                end
            end
        end
    end
endmodule
