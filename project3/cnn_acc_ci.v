// `timescale 1ns / 1ps
// //////////////////////////////////////////////////////////////////////////////////
// // Company: 
// // Engineer: 
// // 
// // Create Date: 2025/07/21 16:09:05
// // Design Name: 
// // Module Name: cnn_acc_ci
// // Project Name: 
// // Target Devices: 
// // Tool Versions: 
// // Description: 
// // 
// // Dependencies: 
// // 
// // Revision:
// // Revision 0.01 - File Created
// // Additional Comments:
// // 
// //////////////////////////////////////////////////////////////////////////////////
// `include "defines_cnn_core.vh"

// // instance만 설정해주는 곳

// module cnn_acc_ci (
//     input clk,
//     input reset_n,

//     // pooling valid
//     input i_in_valid,
//     // 48 개 중 0, 16, 32
//     input [`CO * `OF_BW-1:0] i_in_pooling,

//     output o_ot_valid,
//     output [`CO * `ACC_BW -1:0] o_ot_ci_acc
// );

//     // weight
//     reg        [                   $clog2(16)-1:0] cnt;
//     reg signed [                              7:0] rom         [0:143];
//     reg        [            `CO * `ACC_BW - 1 : 0] ot_ci_acc;
//     wire       [            `CO * `ACC_BW - 1 : 0] w_ot_ci_acc;
//     reg        [            `CO * `ACC_BW - 1 : 0] r_ot_ci_acc;
//     reg                                            r_valid;
//     wire       [                        `CI-1 : 0] w_ot_valid;

//     wire       [                          `CI-1:0] w_in_valid;
//     wire       [`CI * (`MUL_BW + $clog2(3)) - 1:0] w_ot_kernel;

//     initial begin
//         $readmemh("fc1_weights.mem", rom);
//     end



//     always @(posedge clk or negedge reset_n) begin
//         if (!reset_n) begin
//             r_valid <= 0;
//         end else begin
//             r_valid <= &w_ot_valid;
//         end
//     end

//     always @(posedge clk or negedge reset_n) begin
//         if (!reset_n) begin
//             cnt <= 0;
//         end else if (i_in_valid) begin
//             if (cnt == 15) begin
//                 cnt <= 0;
//                 r_valid <= 1;
//             end else begin
//                 //ot_ci_acc <= r_ot_ci_acc;
//                 cnt <= cnt + 1;
//                 r_valid <= 0;
//             end
//         end else begin
//             r_valid   <= 0;
//             ot_ci_acc <= 0;
//         end
//     end

//     genvar mul_inst;
//     generate
//         for (
//             mul_inst = 0; mul_inst < `CI; mul_inst = mul_inst + 1
//         ) begin : gen_mul_inst
//             wire [`CI * `W_BW - 1 : 0] w_cnn_weight = {
//                 rom[mul_inst*48+cnt],
//                 rom[mul_inst*48+16+cnt],
//                 rom[mul_inst*48+32+cnt]
//             };
//             assign w_in_valid[mul_inst] = i_in_valid;
//             cnn_kernal U_cnn_kernal (
//                 .clk(clk),
//                 .reset_n(reset_n),
//                 .i_pooling_valid(w_in_valid[mul_inst]),
//                 .i_pooling(i_in_pooling),
//                 .i_weight(w_cnn_weight),
//                 .o_kernal_valid(w_ot_valid[mul_inst]),
//                 //[`MUL_BW + $clog2(3) - 1: 0]
//                 .o_kernel(w_ot_kernel[mul_inst*(`MUL_BW+$clog2(
//                     3
//                 ))+:(`MUL_BW+$clog2(
//                     3
//                 ))])
//             );
//         end
//     endgenerate

//     // 확인 필요
//     // 3 * 45
//     integer i;
//     always @(*) begin
//         for (i = 0; i < `CI; i = i + 1) begin
//             ot_ci_acc[i*`ACC_BW +: `ACC_BW] = r_ot_ci_acc[i*`ACC_BW +: `ACC_BW] + w_ot_kernel[i * (`MUL_BW + $clog2(
//                 3))+:(`MUL_BW+$clog2(3))];
//         end
//     end


//     assign w_ot_ci_acc = ot_ci_acc;

//     always @(posedge clk or negedge reset_n) begin
//         if (!reset_n) begin
//             r_ot_ci_acc[0+:`CO*`ACC_BW] <= {`CO * `ACC_BW{1'b0}};
//         end else if (&w_ot_valid) begin
//             r_ot_ci_acc[0+:`CO*`ACC_BW] <= w_ot_ci_acc[0+:`CO*`ACC_BW];
//         end
//     end


//     assign o_ot_valid  = r_valid;
//     assign o_ot_ci_acc = ot_ci_acc;
// endmodule
`timescale 1ns / 1ps
`include "defines_cnn_core.vh"

// CI: 입력 채널 수, CO: 출력 채널 수 (3)
// 48개 중에서 (0, 16, 32) 등 특정 오프셋에 대한 데이터만 처리

module cnn_acc_ci(
    input clk,
    input reset_n,

    // pooling valid
    input i_in_valid,
    // pooling 데이터 입력: CO * OF_BW-bits (즉 3채널 * 폭)
    input [`CO * `OF_BW-1:0] i_in_pooling,

    output o_ot_valid,
    output [`CO * `ACC_BW -1:0] o_ot_ci_acc
    );

    // ---- 파라미터 및 내부 신호 ----
    // pooling 입력 16회 카운트
    reg [$clog2(16)-1:0] cnt; 
    // FC layer weight rom
    reg signed [7:0] rom[0:143];
    // 누적 결과
    reg  [`CO * `ACC_BW - 1 : 0] ot_ci_acc;
    reg  [`CO * `ACC_BW - 1 : 0] r_ot_ci_acc;
    wire [`CO * (`MUL_BW + $clog2(3)) - 1:0] w_ot_kernel;
    wire [`CO-1:0] w_ot_valid;
    reg r_valid;

    // --- 가중치 초기화 ---
    initial begin
       $readmemh("fc1_weights.mem", rom);
    end

    // ------- 입력 데이터 처리 -------
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            cnt <= 0;
        end
        else if(i_in_valid) begin
            if(cnt == 15)
                cnt <= 0;
            else
                cnt <= cnt + 1;
        end
    end

    // ------- valid(출력 유효 신호) 처리 -------
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_valid <= 0;
        end
        else if(i_in_valid && cnt == 15) begin
            r_valid <= 1;
        end
        else begin
            r_valid <= 0;
        end
    end

    // ------- Mul 결과 누적 --------
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            r_ot_ci_acc <= 0;
        end
        else if(i_in_valid) begin
            // 누적 결과 저장 (출력 쪽)
            r_ot_ci_acc <= ot_ci_acc;
        end
        else if(r_valid) begin
            // 맨 마지막 valid에서 누적 결과 리셋
            r_ot_ci_acc <= 0;
        end
    end

    // ------- Mul 연산 및 누적 (조합 논리) -------
    integer i;
    always @(*) begin
        for(i = 0; i < `CO; i = i+1) begin
            if(i_in_valid) 
                ot_ci_acc[i*`ACC_BW +: `ACC_BW] = r_ot_ci_acc[i*`ACC_BW +: `ACC_BW] + 
                    w_ot_kernel[i * (`MUL_BW + $clog2(3)) +: (`MUL_BW + $clog2(3))];
            else
                ot_ci_acc[i*`ACC_BW +: `ACC_BW] = r_ot_ci_acc[i*`ACC_BW +: `ACC_BW];
        end
    end

    // ---- Mul Instance 생성 ----
    genvar mul_inst;
    generate
        for(mul_inst = 0; mul_inst < `CO; mul_inst = mul_inst + 1) begin : gen_mul_inst
            // [0,16,32] 오프셋별 weight 뽑기
            wire [7:0] weight0 = rom[mul_inst*48 + cnt];
            wire [7:0] weight1 = rom[mul_inst*48 + 16 + cnt];
            wire [7:0] weight2 = rom[mul_inst*48 + 32 + cnt];
            wire [`CO*`W_BW-1:0] w_cnn_weight = {weight2, weight1, weight0}; // ch2,ch1,ch0

            wire in_valid = i_in_valid;
            cnn_kernal U_cnn_kernal(
                .clk(clk),
                .reset_n(reset_n),
                .i_pooling_valid(in_valid),
                .i_pooling(i_in_pooling),
                .i_weight(w_cnn_weight),
                .o_kernal_valid(w_ot_valid[mul_inst]),
                .o_kernel(w_ot_kernel[mul_inst * (`MUL_BW + $clog2(3)) +: (`MUL_BW + $clog2(3))])
            );
        end
    endgenerate

    assign o_ot_valid = r_valid;
    assign o_ot_ci_acc = ot_ci_acc;

endmodule
