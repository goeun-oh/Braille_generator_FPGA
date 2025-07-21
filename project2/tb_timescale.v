`timescale 1ns / 1ps

// DUT와 동일한 파라미터 파일을 include합니다.
`define IF_BW           32  // 입력 Feature Map 픽셀 비트폭 (Max Pool 입력)
`define OF_BW           32  // 출력 Feature Map 픽셀 비트폭 (Max Pool 출력, FC 입력)
`define W_BW             7  // FC Layer 가중치(Weight) 비트폭
`define BIAS_BW          6  // FC Layer 편향(Bias) 비트폭

// 연산 과정에서의 비트 폭
`define MUL_BW          (`OF_BW + `W_BW)            // 곱셈 결과 비트폭 (32 + 7 = 39)
`define ACC_BW          (`MUL_BW + $clog2(`FC_IN_VEC)) // 내적 누산기 비트폭 (39 + log2(48) -> 39 + 6 = 45)
`define OUT_BW          (`ACC_BW)               // 최종 출력 뉴런 비트폭 (누산기 + Bias 덧셈 후: 45 + 1 = 46)

//----------------------------------------------------------------------
// 2. 레이어 차원 (Layer Dimensions)
//----------------------------------------------------------------------
`define CI               3  // 입력 채널 수 (Input Channels)
`define CO               3  // 출력 채널/뉴런 수 (Output Channels / FC Neurons)

//----------------------------------------------------------------------
// 3. Max Pooling 레이어 관련 파라미터
//----------------------------------------------------------------------
`define POOL_IN_SIZE     8  // Max Pool 입력 Feature Map의 한 변 크기 (8x8)
`define POOL_K           2  // Max Pool 커널 및 스트라이드 크기 (2x2)
`define P_SIZE          (`POOL_IN_SIZE / `POOL_K) // Max Pool 출력 Feature Map 한 변 크기 (4x4)

//----------------------------------------------------------------------
// 4. Fully-Connected (FC) 레이어 관련 파라미터
//----------------------------------------------------------------------
// FC 입력 벡터 길이 (Flatten 후)
`define FC_IN_VEC        (`CI * `P_SIZE * `P_SIZE)   // 3 * 4 * 4 = 48

module tb_matrixmultiplex();

    //======================================================================
    // Testbench Parameters
    //======================================================================
    localparam CLK_PERIOD = 10; // Clock 주기 (10ns -> 100MHz)

    //======================================================================
    // Testbench Signals
    //======================================================================
    // DUT 입력 (reg 타입)
    reg                                      clk;
    reg                                      reset_n;
    reg                                      i_in_valid;
    reg  [`FC_IN_VEC*`OF_BW-1:0]             i_in_fmap;

    // DUT 출력 (wire 타입)
    wire                                    o_ot_valid;
    wire [`CO*`OUT_BW-1:0]                   o_ot_bias;
    
    //======================================================================
    // DUT (Device Under Test) Instantiation
    //======================================================================
    matrixmultiplex dut (
        .clk        (clk),
        .reset_n    (reset_n),
        .i_in_valid (i_in_valid),
        .i_in_fmap  (i_in_fmap),
        .o_ot_valid (o_ot_valid),
        .o_ot_bias  (o_ot_bias)
    );

    //======================================================================
    // Clock Generator
    //======================================================================
    always #(CLK_PERIOD / 2) clk = ~clk;

    //======================================================================
    // Verification Scenario
    //======================================================================
    initial begin
        // --- 1. 초기화 및 리셋 ---
        clk         = 1'b0;
        reset_n     = 1'b1;
        i_in_valid  = 1'b0;
        i_in_fmap   = 'h0;
        $display("[%0t] Simulation Started.", $time);

        #CLK_PERIOD;
        reset_n = 1'b0; // 리셋 활성화 (Active-low)
        $display("[%0t] Reset Asserted.", $time);
        #(CLK_PERIOD * 2);

        reset_n = 1'b1; // 리셋 비활성화
        $display("[%0t] Reset De-asserted. System is ready.", $time);
        #CLK_PERIOD;

        // --- 2. 입력 데이터 생성 및 주입 ---
        generate_input_vector();
        $display("[%0t] Input vector generated. Applying to DUT...", $time);
        
        // 입력 데이터와 함께 valid 신호를 1 클럭 동안 인가
        i_in_valid = 1'b1;
        #CLK_PERIOD;
        i_in_valid = 1'b0;

        // --- 3. 결과 대기 및 확인 (수정된 부분) ---
        // o_ot_valid가 1이 될 때까지 기다립니다.
        wait (o_ot_valid);
        $display("[%0t] Output is valid! Waiting for value to settle...", $time);
        
        // 레이스 컨디션을 피하기 위해 #1의 지연을 주거나,
        // 가장 안정적인 방법인 다음 클럭 엣지까지 기다립니다.
        @(posedge clk);
        
        // --- 4. 최종 출력 값 표시 ---
        $display("[%0t] Displaying results.", $time);
        display_results();

        #(CLK_PERIOD * 5);
        $display("[%0t] Simulation Finished.", $time);
        $finish;
    end

    // --- Task: 입력 벡터 데이터 생성 ---
    // 디버깅 용이성을 위해 0부터 47까지 순차적인 값을 할당
    task generate_input_vector;
        integer i;
        begin
            for (i = 0; i < `FC_IN_VEC; i = i + 1) begin
                i_in_fmap[i*`OF_BW +: `OF_BW] = i;
            end
        end
    endtask

    // --- Task: 최종 출력 결과 표시 ---
    task display_results;
        integer i;
        begin
            $display("--------------------------------------------------");
            $display("Final Output Neuron Values:");
            for (i = 0; i < `CO; i = i + 1) begin
                // OUT_BW에 맞게 sign-extended 된 값을 10진수로 출력
                $display("  - Neuron[%0d]: %d", i, $signed(o_ot_bias[i*`OUT_BW +: `OUT_BW]));
            end
            $display("--------------------------------------------------");
        end
    endtask

endmodule