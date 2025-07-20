`timescale 1ns / 1ps
// 250606
module ImageRom (
    input  logic [9:0] x_pixel,
    input  logic [9:0] y_pixel,
    input  logic       DE,
    output logic [3:0] red_port,
    output logic [3:0] green_port,
    output logic [3:0] blue_port
);
    logic [16:0] image_addr;
    logic [15:0] image_data;  // RGB565 => 16'b rrrrr_gggggg_bbbbb
    //-> 이거를 444로 변경해줘야함. 상위 4비트씩 취하기. 하위비트는 버리기
    // assign image_addr = 320 * y_pixel + x_pixel;  // 640x480 해상도 가정
    // assign image_addr = 320*(y_pixel/2)+(x_pixel/2);
    assign image_addr = 160*(y_pixel/4)+(x_pixel/4);    // qqvga 테스트중

    assign {red_port, green_port, blue_port} = (DE)? {image_data[15:12], image_data[10:7], image_data[4:1]} : 10'b0;
    
    // always_comb begin
    //     if (DE) begin
    //         if (y_pixel > 239 || x_pixel > 319) begin
    //             red_port   = 4'b0;
    //             green_port = 4'b0;
    //             blue_port  = 4'b0;
    //             // DE = 1'b0;
    //         end else begin
    //             red_port   = image_data[15:12];
    //             green_port = image_data[10:7];
    //             blue_port  = image_data[4:1];
    //         end
    //     end else begin
    //             red_port   = 4'b0;
    //             green_port = 4'b0;
    //             blue_port  = 4'b0;
    //         end
    // end
 
    image_rom U_Rom (
        .addr(image_addr),
        .data(image_data)
    );

endmodule

module image_rom (
    input  logic [16:0] addr,
    output logic [15:0] data
);
    // QVGA 해상도 320x240을 가정
    // logic [15:0] rom[0:320*240-1];  // qvga
    logic [15:0] rom[0:160*120-1];  // qqvga

    // 읽은 값을 rom에 저장
    initial begin
        $readmemh("field_qqvga.mem", rom);
        // $readmemh("Lenna.mem", rom);
        // $readmemh("bg2.mem", rom);
    end

    assign data = rom[addr];
endmodule
