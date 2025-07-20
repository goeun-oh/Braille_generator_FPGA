`timescale 1ns / 1ps

module cnn_top #(
    parameter I_F_BW  = 8,
    parameter O_F_BW  = 23,
    parameter KX      = 5,
    parameter KY      = 5,
    parameter W_BW = 7,
    parameter CI      = 1,
    parameter CO      = 3,
    parameter IX   = 28,
    parameter IY   = 28,
    parameter OUT_W   = IX - KX + 1,
    parameter OUT_H   = IY - KY + 1
)(
    input                  clk,
    input                  reset_n,
    input                  i_valid,
    input  [I_F_BW-1 : 0]  i_pixel,
    input  [CO*CI*KX*KY*W_BW-1 : 0] i_cnn_weight,
    input  [CO*7-1 : 0]     i_cnn_bias,
    output reg             o_done
);
    // ===============================
    // cnn_core instance
    // ===============================
    wire                  w_core_valid;
    wire [CO*O_F_BW-1:0]  w_core_fmap;
    //디버깅//
    wire [KX*KY*I_F_BW-1:0] dbg_window;

    cnn_core #(
        .I_F_BW(I_F_BW),
        .KX(KX), .KY(KY),
        .W_BW(7),
        .B_BW(7),
        .CI(CI), .CO(CO),
        .AK_BW(20), .ACI_BW(22),
        .O_F_BW(O_F_BW),
        .AB_BW(24)
    ) u_cnn_core (
        .clk          (clk),
        .reset_n      (reset_n),
        .i_cnn_weight (i_cnn_weight),
        .i_cnn_bias   (i_cnn_bias),
        .i_in_valid   (i_valid),
        .i_in_fmap    (i_pixel),
        .o_ot_valid   (w_core_valid),
        .o_ot_fmap    (w_core_fmap)
    );

    // ===============================
    // Output coordinate counters
    // ===============================
    reg [4:0] x_cnt, y_cnt;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_cnt <= 0;
            y_cnt <= 0;
        end else if (w_core_valid) begin
            if (x_cnt == OUT_W - 1) begin
                x_cnt <= 0;
                if(y_cnt == OUT_H -1) begin
                    y_cnt <= 0;
                end else begin
                    y_cnt <= y_cnt + 1;
                end
            end else begin
                x_cnt <= x_cnt + 1;
            end
        end
    end

    // ===============================
    // Output fmap memory: [CO][24][24]
    // ===============================
    reg [O_F_BW-1:0] result_fmap [0:CO-1][0:OUT_H-1][0:OUT_W-1];

    integer ch;
    always @(posedge clk) begin
        if (w_core_valid) begin
            for (ch = 0; ch < CO; ch = ch + 1) begin
                result_fmap[ch][y_cnt][x_cnt] <= w_core_fmap[ch*O_F_BW +: O_F_BW];
            end
        end
    end

    // ===============================
    // Done signal: after last pixel
    // ===============================
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            o_done <= 0;
        end else if (w_core_valid && (x_cnt == OUT_W-1) && (y_cnt == OUT_H-1)) begin
            o_done <= 1;
        end
    end

endmodule
