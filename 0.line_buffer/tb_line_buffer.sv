`timescale 1ns / 1ps

module tb_line_buffer;

    parameter IX = 28;
    parameter IY = 28;
    parameter KX = 5;
    parameter KY = 5;
    parameter I_F_BW = 8;

    logic clk;
    logic reset_n;
    logic i_in_valid;
    logic [I_F_BW-1:0] i_in_pixel;
    logic o_window_valid;
    logic [KX*KY*I_F_BW-1:0] o_window;
    logic [$clog2(IX)-1:0] o_if_x;
    logic [$clog2(IY)-1:0] o_if_y;

    // Instantiate DUT
    line_buffer #(
        .I_F_BW(I_F_BW),
        .IX(IX),
        .IY(IY),
        .KX(KX),
        .KY(KY)
    ) uut (
        .clk(clk),
        .reset_n(reset_n),
        .i_in_valid(i_in_valid),
        .i_in_pixel(i_in_pixel),
        .o_window_valid(o_window_valid),
        .o_window(o_window),
        .o_if_x(o_if_x),
        .o_if_y(o_if_y)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Input image generation
    logic [I_F_BW-1:0] input_image [0:IX*IY-1];
    initial begin
        for (int i = 0; i < IX*IY; i++) begin
            input_image[i] = i + 1; // 1, 2, 3, ... 784
        end
    end

    // Test sequence
    initial begin
        clk = 0;
        reset_n = 0;
        i_in_valid = 0;
        i_in_pixel = 0;

        #10;
        reset_n = 1;

        for (int i = 0; i < IX*IY; i++) begin
            @(posedge clk);
            i_in_valid <= 1;
            i_in_pixel <= input_image[i];
        end

        // Stop input
        @(posedge clk);
        i_in_valid <= 0;

        // Wait a bit
        $finish;
    end
    // Window print
    int idx;
    always @(posedge clk) begin
        if (o_window_valid) begin
            $display("=== Window at valid ===");
            for (int wy = 0; wy < 5; wy++) begin
                $write("Row %0d: ", wy);
                for (int wx = 0; wx < KX; wx++) begin
                    idx = (wy*KX + wx) * I_F_BW;
                    $write("%0d ", o_window[idx +: I_F_BW]);
                end
                $write("\n");
            end
            $write("\n");
            $write("o_if_x: %0d", o_if_x);
            $write("o_if_y: %0d", o_if_y);
            $write("\n");
            $finish;
        end
    end

endmodule
