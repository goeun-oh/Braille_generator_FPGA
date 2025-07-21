`timescale 1ns / 1ps

module cnn_core #(
    parameter I_F_BW = 8,
    parameter KX = 5,
    parameter KY = 5,
    parameter W_BW = 7,
    parameter B_BW = 7,  //bias
    parameter CI = 1,
    parameter CO = 3,
    parameter AK_BW = 20,
    parameter ACI_BW = 22,
    parameter O_F_BW = 23,
    parameter AB_BW   = 24
) (
    // Clock & Reset
    input                           clk,
    input                           reset_n,
    input  [CO*CI*KX*KY*W_BW-1 : 0] i_cnn_weight,
    input  [         CO*B_BW-1 : 0] i_cnn_bias,
    input                           i_in_valid,
    input  [          I_F_BW-1 : 0] i_in_fmap,
    output                          o_ot_valid,
    output [       CO*O_F_BW-1 : 0] o_ot_fmap,
    //디버깅//
    output [KX*KY*I_F_BW-1:0] o_window

);

    localparam LATENCY = 1;


    //==============================================================================
    // Data Enable Signals 
    //==============================================================================
    wire [LATENCY-1 : 0] ce;
    reg  [LATENCY-1 : 0] r_valid;
    wire [     CO-1 : 0] w_ot_valid;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[LATENCY-1] <= &w_ot_valid;
        end
    end

    assign ce = r_valid;

    //==============================================================================
    // line buffer instance
    //==============================================================================

    wire [KX*KY*I_F_BW-1 : 0] w_window;
    wire                     w_window_valid;

    // CI 채널이므로 line_buffer도 각 채널별로 존재해야 함
    genvar lb_ci;
    generate
    for (lb_ci = 0; lb_ci < CI; lb_ci = lb_ci + 1) begin : gen_line_buffer
        line_buffer #(
        .KX(KX), .KY(KY), .I_F_BW(I_F_BW)
        ) u_line_buffer (
        .clk        (clk),
        .reset_n    (reset_n),
        .i_in_valid    (i_in_valid),
        .i_in_pixel    (i_in_fmap), // 현재는 단일 채널로 가정
        .o_window_valid    (w_window_valid),
        .o_window   (w_window)
        );
    end
    endgenerate
    //디버깅//
    assign o_window = w_window;
    //==============================================================================
    // acc ci instance
    //==============================================================================

    wire [         CO-1 : 0] w_in_valid;
    wire [CO*(ACI_BW)-1 : 0] w_ot_ci_acc;

    // TODO Call cnn_acc_ci Instance
    genvar ci_inst;
    generate
        for (
            ci_inst = 0; ci_inst < CO; ci_inst = ci_inst + 1
        ) begin : gen_ci_inst
            wire    [CI*KX*KY*W_BW-1 : 0]  	w_cnn_weight 	= i_cnn_weight[ci_inst*CI*KY*KX*W_BW +: CI*KY*KX*W_BW];
            cnn_acc_ci u_cnn_acc_ci (
                .clk         (clk),
                .reset_n     (reset_n),
                .i_cnn_weight(w_cnn_weight),
                .i_in_valid  (w_window_valid),
                .i_window (w_window),
                .o_ot_valid  (w_ot_valid[ci_inst]),
                .o_ot_ci_acc (w_ot_ci_acc[ci_inst*(ACI_BW)+:(ACI_BW)])
            );
        end
    endgenerate

    //==============================================================================
    // add_bias = acc + bias
    //==============================================================================
    wire [CO*AB_BW-1 : 0] add_bias;
    reg  [CO*AB_BW-1 : 0] r_add_bias;

    // TODO add bias
    genvar add_idx;
    generate
        for (
            add_idx = 0; add_idx < CO; add_idx = add_idx + 1
        ) begin : gen_add_bias
            assign  add_bias[add_idx*AB_BW +: AB_BW] = w_ot_ci_acc[add_idx*(ACI_BW) +: ACI_BW] + i_cnn_bias[add_idx*B_BW +: B_BW];

            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_add_bias[add_idx*AB_BW+:AB_BW] <= {AB_BW{1'b0}};
                end else if (&w_ot_valid) begin
                    r_add_bias[add_idx*AB_BW +: AB_BW]   <= add_bias[add_idx*AB_BW +: AB_BW];
                end
            end
        end
    endgenerate

    //==============================================================================
    // No Activation
    //==============================================================================
    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_fmap  = r_add_bias;

endmodule

