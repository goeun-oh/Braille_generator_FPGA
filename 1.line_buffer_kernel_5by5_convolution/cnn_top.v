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
    output reg             o_done,
    //디버깅
    output [KX*KY*I_F_BW-1:0] o_window,
    output [KX*I_F_BW-1:0] o_line_buf

);
    // ===============================
    // cnn_core instance
    // ===============================
    wire [CO-1:0] w_core_valid;
    wire [KX*KY*I_F_BW-1:0] w_window;
    assign o_window = w_window;
    wire [CO*O_F_BW-1:0] w_core_fmap;
    genvar co;
    generate
        for (co = 0; co < CO; co = co + 1) begin : gen_cnn_core
            cnn_core #(
                .I_F_BW(I_F_BW),
                .KX(KX), .KY(KY),
                .W_BW(W_BW),
                .B_BW(7),
                .CI(CI), .CO(1),  // CO=1, 각 core는 1개의 출력 채널만 처리
                .AK_BW(20), .ACI_BW(22),
                .O_F_BW(O_F_BW),
                .AB_BW(24)
            ) u_cnn_core (
                .clk(clk),
                .reset_n(reset_n),
                .i_cnn_weight(i_cnn_weight[co*KX*KY*CI*W_BW +: KX*KY*CI*W_BW]),
                .i_cnn_bias(i_cnn_bias[co*7 +: 7]),
                .i_in_valid(i_valid),
                .i_in_fmap(i_pixel),
                .o_ot_valid(w_core_valid[co]),
                .o_ot_fmap(w_core_fmap[co*O_F_BW +: O_F_BW]),
                .o_window(w_window),       // 디버깅용이므로 연결 안 해도 됨
                .o_line_buf()      // 디버깅용이므로 연결 안 해도 됨
            );
        end
    endgenerate

    // ===============================
    // Output coordinate counters
    // ===============================
    reg [4:0] x_cnt, y_cnt;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_cnt <= 0;
            y_cnt <= 0;
        end else if (&w_core_valid) begin
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
        if (&w_core_valid) begin
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
        end else if (&w_core_valid && (x_cnt == OUT_W-1) && (y_cnt == OUT_H-1)) begin
            o_done <= 1;
        end
    end

endmodule
