`timescale 1ns / 1ps

module clk_div5 (
    input  wire clk,        // 100MHz 입력 클럭
    input  wire reset_n,    // 비동기 리셋 (active low)
    output reg  clk_out     // 20MHz 출력 클럭
);

    reg [2:0] cnt;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            cnt     <= 3'd0;
            clk_out <= 1'b0;
        end else begin
            if (cnt == 3'd3) begin
                clk_out <= ~clk_out;
                cnt <= 0;
            end else begin
                cnt <= cnt + 1;
            end
        end
    end

endmodule

