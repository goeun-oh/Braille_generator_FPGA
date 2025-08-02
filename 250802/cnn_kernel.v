`timescale 1ns / 1ps
`include "defines_cnn_core.v"
module cnn_kernel(
    // Clock & Reset
    input clk,
    input reset_n,
    input [`KX*`KY*`ST1_W_BW-1 : 0] i_cnn_weight,
    input i_in_valid,
    input [`KX*`KY*`ST1_I_F_BW-1 : 0] i_in_fmap,
    output o_ot_valid,
    output signed [`ST1_AK_BW-1 : 0] o_ot_kernel_acc
    
);


    localparam LATENCY = 3;


    //==============================================================================
    // Data Enable Signals 
    //==============================================================================
    //shift register
    //입력이 들어오면 r_valid가 1.
    //곱셈 연산에 한 클럭 소모
    //누산(accumulation) 연산에 1클럭 소모
    //총 2클럭의 지연이 있어서 valid 신호도 그만큼 지연되어야함
    wire [LATENCY-1 : 0] ce;
    reg  [LATENCY-1 : 0] r_valid;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[LATENCY-2] <= i_in_valid;
            r_valid[LATENCY-1] <= r_valid[LATENCY-2];
        end
    end

    assign ce = r_valid;

    //==============================================================================
    // mul = fmap * weight
    //==============================================================================

    wire [`KY*`KX*`ST1_I_M_BW-1 : 0] mul;
    reg [`KY*`KX*`ST1_I_M_BW-1 : 0] r_mul;

    // TODO Multiply each of Kernels
    genvar mul_idx;
    generate
        for (
            mul_idx = 0; mul_idx < `KY * `KX; mul_idx = mul_idx + 1
        ) begin : gen_mul
            assign  mul[mul_idx * `ST1_I_M_BW +: `ST1_I_M_BW]   =$signed({1'b0, i_in_fmap[mul_idx * `ST1_I_F_BW +: `ST1_I_F_BW]}) * $signed(i_cnn_weight[mul_idx * `ST1_W_BW +: `ST1_W_BW]);
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    r_mul[mul_idx*`ST1_I_M_BW+:`ST1_I_M_BW] <= 0;
                end else if (i_in_valid) begin
                    r_mul[mul_idx*`ST1_I_M_BW+:`ST1_I_M_BW] <= $signed(mul[mul_idx*`ST1_I_M_BW+:`ST1_I_M_BW]);
                end
            end
        end
    endgenerate
    reg signed [`ST1_I_M_BW-1:0] reg_r_mul [0:`KY-1][0:`KX-1];
    integer k;
    integer j;
    always @(posedge clk) begin
        if(i_in_valid)begin
            for (k= 0; k < `KY; k = k + 1) begin
                for (j= 0; j < `KX; j = j + 1) begin
                    reg_r_mul[k][j] <= $signed(r_mul[(k*`KY+j)*`ST1_I_M_BW +: `ST1_I_M_BW]);
                end
            end
        end
    end

    //(* mark_debug = "true" *) reg signed [`ST1_I_M_BW-1:0] d_reg_mul [0:`KX-1];
    //always @(posedge clk, negedge reset_n) begin
    //    if (!reset_n) begin
    //        d_reg_mul [0] <= 0; 
    //        d_reg_mul [1] <= 0;
    //        d_reg_mul [2] <= 0;
    //        d_reg_mul [3] <= 0;
    //        d_reg_mul [4] <= 0;
    //    end else begin
    //        d_reg_mul [0] <= reg_r_mul[0][0];
    //        d_reg_mul [1] <= reg_r_mul[0][1];
    //        d_reg_mul [2] <= reg_r_mul[0][2];
    //        d_reg_mul [3] <= reg_r_mul[0][3];
    //        d_reg_mul [4] <= reg_r_mul[0][4];
    //    end
    //end

    reg signed [`ST1_AK_BW-1 : 0] acc_kernel;
    reg signed [`ST1_AK_BW-1 : 0] r_acc_kernel;
    integer               acc_idx;
    generate
        always @(*) begin
            acc_kernel[0+:`ST1_AK_BW] = 0;
            for (acc_idx = 0; acc_idx < `KY * `KX; acc_idx = acc_idx + 1) begin
                acc_kernel[0 +: `ST1_AK_BW] = $signed(acc_kernel[0 +: `ST1_AK_BW]) + $signed(r_mul[acc_idx*`ST1_I_M_BW +: `ST1_I_M_BW]);
            end
        end
    endgenerate

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_acc_kernel[0+:`ST1_AK_BW] <= 0;
        end else if (ce[LATENCY-2]) begin
            r_acc_kernel[0+:`ST1_AK_BW] <= $signed(acc_kernel[0+:`ST1_AK_BW]);
        end
    end

    reg [`ST1_W_BW-1:0] reg_weight [0:`KY-1][0:`KX-1];
    always @(posedge clk) begin
        for (k= 0; k < `KY; k = k + 1) begin
            for (j= 0; j < `KX; j = j + 1) begin
                reg_weight[k][j] <= i_cnn_weight[(k*`KY+j)*`ST1_W_BW +: `ST1_W_BW];
            end
        end
    end
    
    reg [`ST1_I_F_BW-1:0] reg_i_fmap [0:`KY-1][0:`KX-1];
    always @(posedge clk) begin
        if(i_in_valid)begin
            for (k= 0; k < `KY; k = k + 1) begin
                for (j= 0; j < `KX; j = j + 1) begin
                    reg_i_fmap[k][j] <= i_in_fmap[(k*`KY+j)*`ST1_I_F_BW +: `ST1_I_F_BW];
                end
            end
        end
    end

    assign o_ot_valid = r_valid[LATENCY-1];
    assign o_ot_kernel_acc = r_acc_kernel;

    
endmodule