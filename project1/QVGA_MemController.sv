`timescale 1ns / 1ps

module QVGA_MemController (
    // VGA Controller side
    input  logic        clk,
    input  logic        pclk,        // 픽클락 제너레이터에 아웃풋 추가해서 pclk라고 표현
    input  logic [ 9:0] x_pixel,
    input  logic [ 9:0] y_pixel,
    input  logic        DE,
    input  logic        upscale,     // upscale 스위치
    input  logic        median,      // 스위치
    // frame buffer side
    output logic        rclk,
    output logic        d_en,
    output logic [16:0] rAddr,
    input  logic [15:0] rData,
    // export side
    output logic [ 3:0] red_port,
    output logic [ 3:0] green_port,
    output logic [ 3:0] blue_port
    // chroma back 전용 upscale
    // output logic [9:0] x_pixel_chroma,
    // output logic [9:0] y_pixel_chroma
);
    logic display_en;
    logic [9:0] x_pixel_m2, y_pixel_m2;

    // assign x_pixel_chroma = x_pixel_m2;
    // assign y_pixel_chroma = y_pixel_m2;

    assign rclk = clk;

    assign {x_pixel_m2, y_pixel_m2} = upscale ? {1'b0, x_pixel[9:1], 1'b0, y_pixel[9:1]} : {x_pixel, y_pixel};
    // assign display_en = mode2 ? (x_pixel < 640 && y_pixel < 480) : (x_pixel < 320 && y_pixel < 240);
    assign display_en = (x_pixel < 640 && y_pixel < 480);
    assign d_en = display_en;
    assign rAddr = display_en ? (y_pixel_m2 * 320 + x_pixel_m2) : 0;

    // Median filter 적용 전 데이터
    logic [11:0] raw_pixel = {rData[15:12], rData[10:7], rData[4:1]};
    logic [11:0] filtered_pixel;


    // 444로 변경
    // RGB565 => 16'b rrrrr_gggggg_bbbbb
    // assign {red_port, green_port, blue_port} = display_en ? {rData[15:12], rData[10:7], rData[4:1]} : 12'b0;

    // === Median Filter 구조 ===
    logic [11:0] line0[0:319];
    logic [11:0] line1[0:319];
    logic [11:0] line2[0:319];
    logic [11:0] window[0:8];

    // Line buffer shift
    always_ff @(posedge pclk) begin
        if (median && display_en) begin
            line2[x_pixel_m2] <= line1[x_pixel_m2];
            line1[x_pixel_m2] <= line0[x_pixel_m2];
            line0[x_pixel_m2] <= raw_pixel;
        end
    end

    // 윈도우 구성
    always_comb begin
        window[0] = line2[(x_pixel_m2>0)?x_pixel_m2-1 : 0];
        window[1] = line2[x_pixel_m2];
        window[2] = line2[(x_pixel_m2<319)?x_pixel_m2+1 : 319];

        window[3] = line1[(x_pixel_m2>0)?x_pixel_m2-1 : 0];
        window[4] = line1[x_pixel_m2];
        window[5] = line1[(x_pixel_m2<319)?x_pixel_m2+1 : 319];

        window[6] = line0[(x_pixel_m2>0)?x_pixel_m2-1 : 0];
        window[7] = line0[x_pixel_m2];
        window[8] = line0[(x_pixel_m2<319)?x_pixel_m2+1 : 319];
    end

    // Median 연산 적용 여부
    always_comb begin
        filtered_pixel = median ? median9(window) : raw_pixel;
    end

    assign {red_port, green_port, blue_port} = display_en ? filtered_pixel : 12'b0;

    // Median 함수
    function logic [11:0] median9(input logic [11:0] win[0:8]);
        logic [11:0] sorted[0:8];
        logic [11:0] tmp;
        integer i, j;
        begin
            for (i = 0; i < 9; i++) sorted[i] = win[i];
            for (i = 0; i < 9; i++) begin
                for (j = 0; j < 8 - i; j++) begin
                    if (sorted[j] > sorted[j+1]) begin
                        tmp = sorted[j];
                        sorted[j] = sorted[j+1];
                        sorted[j+1] = tmp;
                    end
                end
            end
            return sorted[4];
        end
    endfunction


endmodule

