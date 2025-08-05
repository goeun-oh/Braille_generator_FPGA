`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/23 12:35:49
// Design Name: 
// Module Name: top_cnn
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: ST3_W_BW
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
`include "defines_cnn_core.v"


module stage3_top_cnn(
    input wire clk,
    input wire reset_n,

    input wire i_Relu_valid,
    input wire [`stage3_CI * `ST3_IF_BW - 1: 0] i_in_Relu,

    output o_valid,
    output [7:0] alpha,
    output [3:0] led
    );

    wire pool_valid;
    wire [`pool_CO * `ST3_OF_BW-1:0] w_pool;
    wire acc_valid;
    wire [`acc_CO * `ST3_ACC_BW-1:0] w_acc;
    wire core_valid;
    wire [`core_CO * `ST3_OUT_BW -1:0] w_core;
    // 확인 완료
    stage3_max_pooling U_stage3_max_pooling(
    .clk(clk),
    .reset_n(reset_n),
    .i_Relu_valid(i_Relu_valid),
    .i_in_Relu(i_in_Relu),
    .o_ot_valid(pool_valid),
    .o_ot_pool(w_pool)
    );

    stage3_cnn_acc_ci U_stage3_cnn_acc_ci(
    .clk(clk),
    .reset_n(reset_n),
    .i_in_valid(pool_valid),
    .i_in_pooling(w_pool),
    .o_ot_valid(acc_valid),
    .o_ot_ci_acc(w_acc)
    );

    stage3_cnn_core U_stage3_cnn_core(
    .clk(clk),
    .reset_n(reset_n),
    .i_in_valid(acc_valid),
    .o_ot_ci_acc(w_acc),
    .o_ot_valid(core_valid),
    .o_ot_result(w_core)
    );
    
    stage3_compare_alpha U_stage3_compare_alpha(
        .clk(clk),
        .reset_n(reset_n),
        .i_in_valid(core_valid),
        .i_in_core(w_core),
        .alpha(alpha),
        .led(led),
        .o_valid(o_valid)
    );

endmodule
// module stage3_compare_alpha (
//     input clk,
//     input reset_n,
//     input i_in_valid,
//     input [`core_CO * `ST3_OUT_BW -1:0] i_in_core,
//     output reg [7:0] alpha,
//     output reg [2:0] led,
//     output o_valid
// );
//     localparam LATENCY = 1;
//     reg signed [`ST3_OUT_BW - 1:0] c_ot_result [0 : `core_CO-1];

//     reg  signed   [LATENCY - 1 : 0]         r_valid;

//     always @(posedge clk or negedge reset_n) begin
//         if(!reset_n) begin
//             r_valid   <= 0;
//         end else begin
//             r_valid[LATENCY - 1]  <= i_in_valid;
//             // r_valid[LATENCY - 2]  <= i_in_valid;
//             // r_valid[LATENCY - 1]  <= r_valid[LATENCY - 2];
//         end
//     end

//     integer i;
//     always @(posedge clk, negedge reset_n) begin
//         if (!reset_n) begin
//             for (i=0;i<`core_CO;i=i+1) begin
//                 c_ot_result[i] <= 0;
//             end
//         end else if (i_in_valid) begin
//             for (i=0;i<`core_CO;i=i+1) begin
//                 c_ot_result[i] <= $signed(i_in_core [i*`ST3_OUT_BW +: `ST3_OUT_BW]);
//             end
//         end
//     end

//     always @(*) begin
//         if ((c_ot_result[0] >= c_ot_result[1]) && (c_ot_result[0] >= c_ot_result[2])) begin
//             alpha = 8'h61;
//             led = 3'b100;
//         end else if ((c_ot_result[1] >= c_ot_result[0]) && (c_ot_result[1] >= c_ot_result[2])) begin
//             alpha = 8'h62;
//             led = 3'b010;
//         end else begin
//             alpha = 8'h63;
//             led = 3'b001;
//         end
//     end

//     assign o_valid = r_valid[LATENCY - 1];


// endmodule
// a = 0x61 b = 0x62, c = 0x63
module stage3_compare_alpha (
    input                               clk,
    input                               reset_n,
    input                               i_in_valid,
    input [`core_CO * `ST3_OUT_BW -1:0] i_in_core,
    output reg                          o_valid,
    output reg [7:0]                    alpha,
    output reg [3:0]                    led
);
    
    // localparam STAGE = $clog2(`core_CO);

    // 26
    reg [`ST3_OUT_BW-1:0]           data_stage0 [0: `core_CO - 1];
    reg [$clog2(`core_CO)-1 : 0]          index_stage0[0: `core_CO - 1];
    // 13
    wire                             valid_stage0[0: (`core_CO/2) - 1];

    // 13
    wire [`ST3_OUT_BW-1:0]           data_stage1 [0: (`core_CO/2)-1];
    wire [$clog2(`core_CO)-1 : 0]    index_stage1[0: (`core_CO/2)-1];
    // 6
    wire                             valid_stage1[0: (`core_CO/4)-1];

    // 7
    wire [`ST3_OUT_BW-1:0]           data_stage2 [0:`core_CO/4];
    wire [$clog2(`core_CO)-1 : 0]    index_stage2[0:`core_CO/4];
    //3
    wire                             valid_stage2[0:`core_CO/8];

    //4
    wire [`ST3_OUT_BW-1:0]           data_stage3 [0:`core_CO/8+1];
    wire [$clog2(`core_CO)-1 : 0]    index_stage3[0:`core_CO/8+1];
    //2
    wire                             valid_stage3[0:(`core_CO/13)-1];

    //2
    wire [`ST3_OUT_BW-1:0]           data_stage4 [0:(`core_CO/13)-1];
    wire [$clog2(`core_CO)-1 : 0]    index_stage4[0:(`core_CO/13)-1];
    //1
    wire                             valid_stage4;
    wire [`ST3_OUT_BW-1:0]           data_stage5;
    wire [$clog2(`core_CO)-1 : 0]    index_stage5;
    
    reg [7:0] o_alpha; 
    reg [3:0] o_led;

    reg start_valid;
    integer i;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            start_valid <= 0;
            o_valid <= 0;
            alpha <= 0;
            led <= 0;
            for (i = 0;i < `core_CO ; i= i + 1) begin
                index_stage0[i] <= i;
                data_stage0[i]  <= 0;
            end
        end else begin
            start_valid <= i_in_valid;
            if (i_in_valid) begin
                for (i = 0; i < `core_CO; i = i + 1) begin
                    data_stage0[i] <= i_in_core[i*`ST3_OUT_BW+:`ST3_OUT_BW];
                end
            end
            o_valid <= valid_stage4;
            alpha <= o_alpha;
            led <= o_led;
        end
    end
    always @(*) begin
        o_alpha = 8'h61;    // 기본값 또는 0
        o_led = 4'b0000;    // 기본값 또는 0
        if (valid_stage4) begin
            case (index_stage5)
                5'd0: begin
                    o_alpha = 8'h61;
                    o_led = 4'b0000;
                end
                5'd1: begin
                    o_alpha = 8'h62;
                    o_led = 4'b0001;
                end
                5'd2: begin
                    o_alpha = 8'h63;
                    o_led = 4'b0010;
                end
                5'd3: begin
                    o_alpha = 8'h64;
                    o_led = 4'b0011;
                end
                5'd4: begin
                    o_alpha = 8'h65;
                    o_led = 4'b0100;
                end
                5'd5: begin
                    o_alpha = 8'h66;
                    o_led = 4'b0101;
                end
                5'd6: begin
                    o_alpha = 8'h67;
                    o_led = 4'b0110;
                end
                5'd7: begin
                    o_alpha = 8'h68;
                    o_led = 4'b0111;
                end
                5'd8: begin
                    o_alpha = 8'h69;
                    o_led = 4'b1000;
                end
                5'd9: begin
                    o_alpha = 8'h6A;
                    o_led = 4'b1001;
                end
                5'd10: begin
                    o_alpha = 8'h6B;
                    o_led = 4'b1010;
                end
                5'd11: begin
                    o_alpha = 8'h6C;
                    o_led = 4'b1011;
                end
                5'd12: begin
                    o_alpha = 8'h6D;
                    o_led = 4'b1100;
                end
                5'd13: begin
                    o_alpha = 8'h6E;
                    o_led = 4'b1101;
                end
                5'd14: begin
                    o_alpha = 8'h6F;
                    o_led = 4'b1110;
                end
                5'd15: begin
                    o_alpha = 8'h70;
                    o_led = 4'b1111;
                end
                5'd16: begin
                    o_alpha = 8'h71;
                    o_led = 4'b0000;
                end
                5'd17: begin
                    o_alpha = 8'h72;
                    o_led = 4'b0000;
                end
                5'd18: begin
                    o_alpha = 8'h73;
                    o_led = 4'b0000;
                end
                5'd19: begin
                    o_alpha = 8'h74;
                    o_led = 4'b0000;
                end
                5'd20: begin
                    o_alpha = 8'h75;
                    o_led = 4'b0000;
                end
                5'd21: begin
                    o_alpha = 8'h76;
                    o_led = 4'b0000;
                end
                5'd22: begin
                    o_alpha = 8'h77;
                    o_led = 4'b0000;
                end
                5'd23: begin
                    o_alpha = 8'h78;
                    o_led = 4'b0000;
                end
                5'd24: begin
                    o_alpha = 8'h79;
                    o_led = 4'b0000;
                end
                5'd25: begin
                    o_alpha = 8'h7A;
                    o_led = 4'b0000;
                end
                default: begin
                    o_alpha = 8'h61;
                    o_led = 4'b0000;
                end
            endcase
        end
    end

    // 13번
    genvar stage0;
    generate
        for(stage0 = 0 ; stage0 < 13 ; stage0 = stage0 + 1) begin
            compare U_compare0(
                .clk(clk),
                .reset_n(reset_n),
                .i_in_valid(start_valid),
                .in_core0(data_stage0[2*stage0]),
                .in_core0_index(index_stage0[2*stage0]),
                .in_core1(data_stage0[2*stage0+1]),
                .in_core1_index(index_stage0[2*stage0+1]),
                .o_ot_valid(valid_stage0[stage0]),
                .out_core(data_stage1[stage0]),
                .o_core_index(index_stage1[stage0])
            );
        end
    endgenerate
    // 잘 넣음

    // 6 13번째는 assign으로 보내기
    genvar stage1;
    generate
        for(stage1 = 0 ; stage1 < 6 ; stage1 = stage1 + 1) begin
            compare U_compare1(
                .clk(clk),
                .reset_n(reset_n),
                .i_in_valid(valid_stage0[2*stage1] & valid_stage0[2*stage1+1]),
                .in_core0(data_stage1[2*stage1]),
                .in_core0_index(index_stage1[2*stage1]),
                .in_core1(data_stage1[2*stage1+1]),
                .in_core1_index(index_stage1[2*stage1+1]),
                .o_ot_valid(valid_stage1[stage1]),
                .out_core(data_stage2[stage1]),
                .o_core_index(index_stage2[stage1])
            );
        end
    endgenerate
    assign data_stage2[6] = data_stage1[12];
    assign index_stage2[6] = index_stage1[12];
    // 잘 넣음

    // 3 7번째는 assign으로 보내기
    genvar stage2;
    generate
        for(stage2 = 0 ; stage2 < 3 ; stage2 = stage2 + 1) begin
            compare U_compare2(
                .clk(clk),
                .reset_n(reset_n),
                .i_in_valid(valid_stage1[2*stage2] & valid_stage1[2*stage2+1]),
                .in_core0(data_stage2[2*stage2]),
                .in_core0_index(index_stage2[2*stage2]),
                .in_core1(data_stage2[2*stage2+1]),
                .in_core1_index(index_stage2[2*stage2+1]),
                .o_ot_valid(valid_stage2[stage2]),
                .out_core(data_stage3[stage2]),
                .o_core_index(index_stage3[stage2])
            );
        end
    endgenerate

    assign data_stage3[3] = data_stage2[6];
    assign index_stage3[3] = index_stage2[6];

    compare U_compare3(
        .clk(clk),
        .reset_n(reset_n),
        .i_in_valid(valid_stage2[0] & valid_stage2[1]),
        .in_core0(data_stage3[0]),
        .in_core0_index(index_stage3[0]),
        .in_core1(data_stage3[1]),
        .in_core1_index(index_stage3[1]),
        .o_ot_valid(valid_stage3[0]),
        .out_core(data_stage4[0]),
        .o_core_index(index_stage4[0])
    );
    compare U_compare4(
        .clk(clk),
        .reset_n(reset_n),
        .i_in_valid(valid_stage2[2]),
        .in_core0(data_stage3[2]),
        .in_core0_index(index_stage3[2]),
        .in_core1(data_stage3[3]),
        .in_core1_index(index_stage3[3]),
        .o_ot_valid(valid_stage3[1]),
        .out_core(data_stage4[1]),
        .o_core_index(index_stage4[1])
    );
    // 1
    compare U_compare5(
        .clk(clk),
        .reset_n(reset_n),
        .i_in_valid(valid_stage3[0] & valid_stage3[1]),
        .in_core0(data_stage4[0]),
        .in_core0_index(index_stage4[0]),
        .in_core1(data_stage4[1]),
        .in_core1_index(index_stage4[1]),
        .o_ot_valid(valid_stage4),
        .out_core(data_stage5),
        .o_core_index(index_stage5)
    );
endmodule

module compare (
    input                               clk,
    input                               reset_n,
    input                               i_in_valid,
    input signed [`ST3_OUT_BW - 1 : 0]  in_core0,
    input [$clog2(`core_CO)-1 : 0]      in_core0_index,
    input signed [`ST3_OUT_BW - 1 : 0]  in_core1,
    input [$clog2(`core_CO)-1 : 0]      in_core1_index,
    output reg                          o_ot_valid,
    output reg [`ST3_OUT_BW - 1 : 0]    out_core,
    output reg [$clog2(`core_CO)-1 : 0] o_core_index
);

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            o_ot_valid <= 1'b0;
            out_core <= 0;
            o_core_index <= 0;
        end else begin
            o_ot_valid <= i_in_valid;
            if (i_in_valid) begin
                if(in_core0 >= in_core1) begin
                    out_core     <= in_core0;
                    o_core_index <= in_core0_index;
                end else begin
                    out_core     <= in_core1;
                    o_core_index <= in_core1_index;
                end    
            end
        end
    end
    
endmodule