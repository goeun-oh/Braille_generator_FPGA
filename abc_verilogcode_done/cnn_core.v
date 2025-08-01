`timescale 1ns / 1ps

module cnn_core #(
    parameter I_F_BW = 8,
    parameter KX = 5,
    parameter KY = 5,
    parameter IX = 28,
    parameter W_BW = 8,
    parameter B_BW = 16,  //bias
    parameter CI = 1,
    parameter CO = 3,
    parameter AK_BW = 21,
    parameter ACI_BW = 21,
    parameter O_F_BW = 20,
    parameter AB_BW   = 21,
    parameter AR_BW = 20
) (
    // Clock & Reset
    input                           clk,
    input                           reset_n,
    input [CO*CI*KX*KY*W_BW-1 : 0] i_cnn_weight,
    input [         CO*B_BW-1 : 0] i_cnn_bias,
    input                           i_in_valid,
    input  [          I_F_BW-1 : 0] i_in_fmap,
    output                          o_ot_valid,
    output [       CO*O_F_BW-1 : 0] o_ot_fmap
    ////디버깅//
    //output [KX*KY*I_F_BW-1:0] o_window,
    //output [KX*I_F_BW-1:0] o_line_buf
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

    //디버깅
    //wire [KX*I_F_BW-1:0] w_line_buf;
    //assign o_line_buf = w_line_buf;
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
    //assign o_window = w_window;
    //assign o_line_buf = w_line_buf;
    //==============================================================================
    // acc ci instance
    //==============================================================================

    wire [         CO-1 : 0] w_in_valid;
    wire signed [CO*(ACI_BW)-1 : 0] w_ot_kernel_acc;
    // TODO Call cnn_acc_ci Instance
    genvar co_inst;
    generate
        for (
            co_inst = 0; co_inst < CO; co_inst = co_inst + 1
        ) begin : gen_co_kernel
            wire signed   [KX*KY*W_BW-1 : 0]  	w_cnn_weight 	= i_cnn_weight[co_inst*KY*KX*W_BW +: KY*KX*W_BW];
            cnn_kernel u_cnn_kernel (
                .clk         (clk),
                .reset_n     (reset_n),
                .i_cnn_weight(w_cnn_weight),
                .i_in_valid  (w_window_valid),
                .i_in_fmap (w_window),
                .o_ot_valid  (w_ot_valid[co_inst]),
                .o_ot_kernel_acc (w_ot_kernel_acc[co_inst*(ACI_BW)+:(ACI_BW)])
            );
        end
    endgenerate

    //==============================================================================
    // add_bias = acc + bias
    //==============================================================================
    //디버깅
    reg signed [CO*AR_BW-1 : 0] r_add_bias;

    // TODO add bias
    genvar add_idx;
    generate
        for (
            add_idx = 0; add_idx < CO; add_idx = add_idx + 1
        ) begin : gen_add_bias
            wire signed [AB_BW-1:0] bias_sum;
            assign bias_sum = $signed(w_ot_kernel_acc[add_idx*(ACI_BW) +: ACI_BW]) + $signed(i_cnn_bias[add_idx*B_BW +: B_BW]);
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_add_bias[add_idx*AR_BW+:AR_BW] <= 0;
                end else if (&w_ot_valid) begin
                    r_add_bias[add_idx*AR_BW +: AR_BW] <= (bias_sum >=0)? bias_sum[AR_BW:0] : 0;
                end
            end
        end
    endgenerate

    //==============================================================================
    // No Activation
    //==============================================================================
    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_fmap  = r_add_bias;
    
    integer ch;
    integer j;
    integer k;
    reg signed [ACI_BW-1:0] reg_ex_bias[0:CO-1];
    always @(posedge clk) begin
        for (ch=0; ch<CO; ch=ch+1)begin
            reg_ex_bias[ch] <= $signed(w_ot_kernel_acc[ch*ACI_BW +: ACI_BW]);
        end
    end
endmodule