`timescale 1ns / 1ps

// DUT와 동일한 파라미터 파일을 include하여 일관성을 유지합니다.
`include "defines_cnn_core.vh"

module tb_pooling_core();

    //======================================================================
    // Testbench Parameters
    //======================================================================
    localparam CLK_PERIOD = 10; // Clock 주기 (10ns -> 100MHz)

    //======================================================================
    // Testbench Signals
    //======================================================================
    // DUT 입력 (reg 타입)
    reg                                                 clk;
    reg                                                 reset_n;
    reg                                                 i_in_valid;
    reg  [`CI*`POOL_IN_SIZE*`POOL_IN_SIZE*`IF_BW-1:0]    i_in_fmap;

    // DUT 출력 (wire 타입)
    wire                                                o_ot_valid;
    wire [`CI*`P_SIZE*`P_SIZE*`OF_BW-1:0]                o_ot_fmap;

    //======================================================================
    // DUT (Device Under Test) Instantiation
    //======================================================================
    max_pool dut(
        .clk        (clk),
        .reset_n    (reset_n),
        .i_in_valid (i_in_valid),
        .i_in_fmap  (i_in_fmap),
        .o_ot_valid (o_ot_valid),
        .o_ot_ci_acc(o_ot_fmap)
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

        // --- 2. 예측 가능한 입력 데이터 생성 및 주입 ---
        generate_predictable_input();
        $display("[%0t] Predictable input generated. Applying to DUT...", $time);
        
        // 입력 데이터와 함께 valid 신호를 1 클럭 동안 인가
        i_in_valid = 1'b1;
        #CLK_PERIOD;
        i_in_valid = 1'b0;

        // --- 3. 결과 대기 및 확인 ---
        // max_pool 모듈의 Latency는 1 사이클이므로, valid 출력을 기다립니다.
        wait (o_ot_valid);
        $display("[%0t] Output is valid!", $time);
        
        // --- 4. 입력 및 출력 값 표시 ---
        display_io_maps();
        
        #(CLK_PERIOD * 5);
        $display("[%0t] Simulation Finished.", $time);
        $finish;
    end
    
    // --- Task: 예측 가능한 입력 데이터 생성 ---
    // 디버깅이 쉽도록 랜덤 값 대신 0부터 순차적으로 증가하는 값을 사용합니다.
    task generate_predictable_input;
        integer ch, i;
        begin
            for (ch = 0; ch < `CI; ch = ch + 1) begin
                for (i = 0; i < `POOL_IN_SIZE * `POOL_IN_SIZE; i = i + 1) begin
                    // 각 채널별로 0~63, 64~127, 128~191 값이 할당됩니다.
                    i_in_fmap[(ch*`POOL_IN_SIZE*`POOL_IN_SIZE + i)*`IF_BW +: `IF_BW] = (ch * 64) + i;
                end
            end
        end
    endtask

    // --- Task: 입출력 결과 비교를 위해 화면에 표시 ---
    task display_io_maps;
        integer ch, r, c, idx;
    begin
        // 입력 Feature Map 표시 (3-channel, 8x8)
        $display("\n===================== INPUT FEATURE MAP (3 x 8 x 8) =====================");
        for (ch = 0; ch < `CI; ch = ch + 1) begin
            $display("--- Channel %0d ---", ch);
            for (r = 0; r < `POOL_IN_SIZE; r = r + 1) begin
                $write("    ");
                for (c = 0; c < `POOL_IN_SIZE; c = c + 1) begin
                    idx = (ch*`POOL_IN_SIZE*`POOL_IN_SIZE) + (r*`POOL_IN_SIZE) + c;
                    $write("%4d ", $signed(i_in_fmap[idx*`IF_BW +: `IF_BW]));
                    if ((c % 2) == 1) $write("| "); // 2x2 영역 구분을 위한 구분선
                end
                $write("\n");
                if ((r % 2) == 1) $display("    -----------------------------------------");
            end
        end

        // 출력 Feature Map 표시 (3-channel, 4x4)
        $display("\n==================== OUTPUT FEATURE MAP (3 x 4 x 4) =====================");
        for (ch = 0; ch < `CI; ch = ch + 1) begin
            $display("--- Channel %0d ---", ch);
            for (r = 0; r < `P_SIZE; r = r + 1) begin
                $write("    ");
                for (c = 0; c < `P_SIZE; c = c + 1) begin
                    idx = (ch*`P_SIZE*`P_SIZE) + (r*`P_SIZE) + c;
                    // DUT의 출력(`o_ot_fmap`)을 올바른 크기와 비트 폭으로 표시합니다.
                    $write("%4d ", $signed(o_ot_fmap[idx*`OF_BW +: `OF_BW]));
                end
                $write("\n");
            end
        end
        $display("=======================================================================\n");
    end
    endtask

endmodule
