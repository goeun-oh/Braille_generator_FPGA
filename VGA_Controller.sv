`timescale 1ns / 1ps

module VGA_Controller (
    input  logic       clk,
    input  logic       reset,
    output logic       rclk,
    output logic       h_sync,
    output logic       v_sync,
    output logic       DE,
    output logic [9:0] x_pixel,
    output logic [9:0] y_pixel
);

    logic pclk;
    logic [9:0] h_counter;
    logic [9:0] v_counter;

    assign rclk = clk; // 그대로 나가기~

    pix_clk_gen U_pix_clk_gen (
        .clk  (clk),
        .reset(reset),
        .pclk (pclk)
    );

    pixel_counter U_pixel_counter (
        .pclk     (pclk),
        .reset    (reset),
        .h_counter(h_counter),
        .v_counter(v_counter)
    );

    vga_decoder U_vga_decoder (
        .h_counter(h_counter),
        .v_counter(v_counter),
        .h_sync   (h_sync),
        .v_sync   (v_sync),
        .x_pixel  (x_pixel),
        .y_pixel  (y_pixel),
        .DE       (DE)
    );

endmodule

module pix_clk_gen (
    input  logic clk,
    input  logic reset,
    output logic pclk
);
    logic [1:0] p_counter;

    always @(posedge clk, posedge reset) begin
        if (reset) begin
            p_counter <= 0;
            pclk <= 1'b0;
        end else begin
            if (p_counter == 3) begin
                p_counter <= 0;
                pclk      <= 1'b1;
            end else begin
                p_counter <= p_counter + 1;
                pclk      <= 1'b0;
            end
        end
    end
endmodule

module pixel_counter (
    input  logic       pclk,
    input  logic       reset,
    output logic [9:0] h_counter,
    output logic [9:0] v_counter
);
    localparam H_MAX = 800, V_MAX = 525;

    always @(posedge pclk, posedge reset) begin
        if (reset) begin
            h_counter <= 0;
        end else begin
            if (h_counter == H_MAX - 1) begin
                h_counter <= 0;
            end else begin
                h_counter <= h_counter + 1;
            end
        end
    end

    always @(posedge pclk, posedge reset) begin
        if (reset) begin
            v_counter <= 0;
        end else begin
            if (h_counter == H_MAX - 1) begin
                if (v_counter == V_MAX - 1) begin
                    v_counter <= 0;
                end else begin
                    v_counter <= v_counter + 1;
                end
            end
        end
    end
endmodule

module vga_decoder (
    input  logic [9:0] h_counter,
    input  logic [9:0] v_counter,
    output logic       h_sync,
    output logic       v_sync,
    output logic [9:0] x_pixel,
    output logic [9:0] y_pixel,
    output logic       DE
);
    localparam H_Visible_area = 640;
    localparam H_Front_porch = 16;
    localparam H_Sync_pulse = 96;
    localparam H_Back_porch = 48;
    localparam H_Whole_line = 800;

    localparam V_Visible_area = 480;
    localparam V_Front_porch = 10;
    localparam V_Sync_pulse = 2;
    localparam V_Back_porch = 33;
    localparam V_Whole_frame = 525;

    assign h_sync = !((h_counter >= (H_Visible_area + H_Front_porch)) 
    && (h_counter < (H_Visible_area + H_Front_porch + H_Sync_pulse)));

    assign v_sync = !((v_counter >= (V_Visible_area + V_Front_porch)) 
    && (v_counter < (V_Visible_area + V_Front_porch + V_Sync_pulse)));

    assign DE = (h_counter < H_Visible_area) && (v_counter < V_Visible_area);

    assign x_pixel = h_counter;
    assign y_pixel = v_counter;


endmodule
