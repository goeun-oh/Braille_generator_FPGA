`timescale 1ns / 1ps
module cnn_acc_ci #(
    parameter I_F_BW = 8,
    parameter KX = 5,
    parameter KY = 5,
    parameter W_BW = 8,
    parameter CI = 1,
    parameter AK_BW = 21,
    parameter ACI_BW = 21
) (
    // Clock & Reset
    input                      clk,
    input                      reset_n,
    input signed [CI*KX*KY*W_BW-1:0] i_cnn_weight,
    input                      i_in_valid,
    input  [ KX*KY*I_F_BW-1:0] i_window,
    output                     o_ot_valid,
    output signed [       ACI_BW-1:0] o_ot_ci_acc    // CI개의 channel 누산 결과
);

    localparam LATENCY = 2;

    //==============================================================================
    // Data Enable Signals 
    //==============================================================================
    wire [LATENCY-1 : 0] ce;
    reg  [LATENCY-1 : 0] r_valid;
    wire [     CI-1 : 0] w_ot_valid;

    assign ce = r_valid;
    //==============================================================================
    // mul_acc kenel instance
    //==============================================================================

    wire [CI*AK_BW-1 : 0] w_ot_kernel_acc;
    wire [  ACI_BW-1 : 0] w_ot_ci_acc;
    reg [  ACI_BW-1 : 0] r_ot_ci_acc;
    genvar mul_inst;
    generate
        for (
            mul_inst = 0; mul_inst < CI; mul_inst = mul_inst + 1
        ) begin : gen_mul_inst
            wire    [KX*KY*W_BW-1 : 0]  	w_cnn_weight 	= i_cnn_weight[mul_inst*KY*KX*W_BW +: KY*KX*W_BW];
            cnn_kernel u_cnn_kernel (
                .clk            (clk),
                .reset_n        (reset_n),
                .i_cnn_weight   (w_cnn_weight),
                .i_in_valid     (i_in_valid),
                .i_in_fmap      (i_window),
                .o_ot_valid     (w_ot_valid[mul_inst]),
                .o_ot_kernel_acc(w_ot_kernel_acc[mul_inst*AK_BW+:AK_BW])
            );
        end
    endgenerate

    reg [ACI_BW-1 : 0]    ot_ci_acc;
    integer i;
    always @(*) begin
        ot_ci_acc = {ACI_BW{1'b0}};
        for (i = 0; i < CI; i = i + 1) begin
            ot_ci_acc = ot_ci_acc + w_ot_kernel_acc[i*AK_BW+:AK_BW];
        end
    end

    //assign w_ot_ci_acc = w_ot_kernel_acc[0*AK_BW +: AK_BW] + w_ot_kernel_acc[(0+1)*AK_BW +: AK_BW];
    assign w_ot_ci_acc = ot_ci_acc;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_ot_ci_acc[0+:ACI_BW] <= 0;
        end else if (&w_ot_valid) begin
            r_ot_ci_acc[0+:ACI_BW] <= w_ot_ci_acc[0+:ACI_BW];
        end
    end
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[LATENCY-1] <= &w_ot_valid;  // shift right, insert new at LSB
        end
    end
    assign o_ot_valid  =r_valid[LATENCY-1];
    assign o_ot_ci_acc = r_ot_ci_acc;

endmodule
 