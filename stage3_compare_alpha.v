`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/04 10:06:12
// Design Name: 
// Module Name: stage3_compare_alpha
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
`include "defines_cnn_core.v"

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

    reg start_valid;
    integer i;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
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
            if (valid_stage4) begin
                case (index_stage5)
                    5'd0: begin
                        alpha <= 8'h61;
                        led <= 4'b0000;
                    end
                    5'd1: begin
                        alpha <= 8'h62;
                        led <= 4'b0001;
                    end
                    5'd2: begin
                        alpha <= 8'h63;
                        led <= 4'b0010;
                    end
                    5'd3: begin
                        alpha <= 8'h64;
                        led <= 4'b0011;
                    end
                    5'd4: begin
                        alpha <= 8'h65;
                        led <= 4'b0100;
                    end
                    5'd5: begin
                        alpha <= 8'h66;
                        led <= 4'b0101;
                    end
                    5'd6: begin
                        alpha <= 8'h67;
                        led <= 4'b0110;
                    end
                    5'd7: begin
                        alpha <= 8'h68;
                        led <= 4'b0111;
                    end
                    5'd8: begin
                        alpha <= 8'h69;
                        led <= 4'b1000;
                    end
                    5'd9: begin
                        alpha <= 8'h6A;
                        led <= 4'b1001;
                    end
                    5'd10: begin
                        alpha <= 8'h6B;
                        led <= 4'b1010;
                    end
                    5'd11: begin
                        alpha <= 8'h6C;
                        led <= 4'b1011;
                    end
                    5'd12: begin
                        alpha <= 8'h6D;
                        led <= 4'b1100;
                    end
                    5'd13: begin
                        alpha <= 8'h6E;
                        led <= 4'b1101;
                    end
                    5'd14: begin
                        alpha <= 8'h6F;
                        led <= 4'b1110;
                    end
                    5'd15: begin
                        alpha <= 8'h70;
                        led <= 4'b1111;
                    end
                    5'd16: begin
                        alpha <= 8'h71;
                    end
                    5'd17: begin
                        alpha <= 8'h72;
                    end
                    5'd18: begin
                        alpha <= 8'h73;
                    end
                    5'd19: begin
                        alpha <= 8'h74;
                    end
                    5'd20: begin
                        alpha <= 8'h75;
                    end
                    5'd21: begin
                        alpha <= 8'h76;
                    end
                    5'd22: begin
                        alpha <= 8'h77;
                    end
                    5'd23: begin
                        alpha <= 8'h78;
                    end
                    5'd24: begin
                        alpha <= 8'h79;
                    end
                    5'd25: begin
                        alpha <= 8'h7A;
                    end
                    default: begin
                        alpha <= 8'h61;
                        led <= 4'b0000;
                    end
                endcase
            end
        end
    end


    // 13번
    genvar stage0;
    generate
        for(stage0 = 0 ; stage0 < (`core_CO/2) ; stage0 = stage0 + 1) begin
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
        for(stage1 = 0 ; stage1 < (`core_CO/4) ; stage1 = stage1 + 1) begin
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
        for(stage2 = 0 ; stage2 < (`core_CO/8) ; stage2 = stage2 + 1) begin
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

    // 2
    genvar stage3;
    generate
        for(stage3 = 0 ; stage3 < (`core_CO/13) ; stage3 = stage3 + 1) begin
            compare U_compare3(
                .clk(clk),
                .reset_n(reset_n),
                .i_in_valid(valid_stage2[2*stage3] & valid_stage2[2*stage3+1]),
                .in_core0(data_stage3[2*stage3]),
                .in_core0_index(index_stage3[2*stage3]),
                .in_core1(data_stage3[2*stage3+1]),
                .in_core1_index(index_stage3[2*stage3+1]),
                .o_ot_valid(valid_stage3[stage3]),
                .out_core(data_stage4[stage3]),
                .o_core_index(index_stage4[stage3])
            );
        end
    endgenerate
    // 1
    compare U_compare4(
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
    input [`ST3_OUT_BW - 1 : 0]         in_core0,
    input [$clog2(`core_CO)-1 : 0]      in_core0_index,
    input [`ST3_OUT_BW - 1 : 0]         in_core1,
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
            if(in_core0 >= in_core1) begin
                out_core     <= in_core0;
                o_core_index <= in_core0_index;
            end else begin
                out_core     <= in_core1;
                o_core_index <= in_core1_index;
            end
        end
    end
    
endmodule