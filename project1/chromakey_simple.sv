`timescale 1ns / 1ps
// 250606
module chromakey_simple (
    input  logic       clk,
    input  logic       reset,
    input  logic       de,
    input  logic       mode_chroma,   // 크로마키 활성화 스위치
    input  logic [3:0] red_mem,
    input  logic [3:0] green_mem,
    input  logic [3:0] blue_mem,
    input  logic [3:0] red_back, // 배경 이미지 픽셀
    input  logic [3:0] green_back,
    input  logic [3:0] blue_back,
    output logic [3:0] red_chroma,
    output logic [3:0] green_chroma,
    output logic [3:0] blue_chroma
);


    // 크로마키 색상 판단: 초록이 우세하면 배경색으로 간주
    logic target_color;
    // assign target_color = (green_mem > red_mem) && (green_mem > blue_mem); // 필터 추가 전에 원본
    assign target_color = (green_mem > red_mem + 2) && (green_mem > blue_mem + 2); // 이거 좀 잘됨
    // assign target_color = (red_mem <= 2) && (green_mem <= 2) && (blue_mem <= 2); // 검정색이 타겟

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            red_chroma   <= 4'd0;
            green_chroma <= 4'd0;
            blue_chroma  <= 4'd0;
        end else if (de) begin
            if (mode_chroma && target_color) begin
                red_chroma   <= red_back;
                green_chroma <= green_back;
                blue_chroma  <= blue_back;
            end else begin
                red_chroma   <= red_mem;
                green_chroma <= green_mem;
                blue_chroma  <= blue_mem;
            end
        end else begin
            red_chroma   <= 4'd0;
            green_chroma <= 4'd0;
            blue_chroma  <= 4'd0;
        end
    end

endmodule
