// tb_top_cnn.v

// 테스트 대상 파일들을 포함합니다.
`include "defines_cnn_core.vh"
// `include "top_cnn.v" // 설계 코드가 포함된 파일명

module tb_top_cnn();

    //======================================================================
    // Testbench Parameters
    //======================================================================
    localparam CLK_PERIOD = 10; // Clock 주기 (10ns -> 100MHz)

    //======================================================================
    // Testbench Signals
    //======================================================================
    // DUT 입력 (reg 타입)
    reg                                                 clk;
    reg                                             reset_n;
    reg                                             i_in_valid;
    reg [`CI*`POOL_IN_SIZE*`POOL_IN_SIZE*`IF_BW-1:0] i_in_fmap;

    // DUT 출력 (wire 타입)
    wire                                        o_ot_valid;
    wire [`CO*`OUT_BW-1:0]                       o_ot_bias;

    //======================================================================
    // DUT (Device Under Test) Instantiation
    //======================================================================
    top_cnn DUT (
        .clk(clk),
        .reset_n(reset_n),
        .i_in_valid(i_in_valid),
        .i_in_fmap(i_in_fmap),
        .o_ot_valid(o_ot_valid),
        .o_ot_bias(o_ot_bias)
    );

    //======================================================================
    // Clock Generator
    //======================================================================
    always #5 clk = ~clk;

    //======================================================================
    // Verification Scenario
    //======================================================================
    initial begin
        // --- 1. 초기화 및 리셋 ---
        clk        = 1'b0;
        reset_n    = 1'b1;
        i_in_valid = 1'b0;
        i_in_fmap  = 'h0;
        $display("[%0t] Simulation Started.", $time);

        #CLK_PERIOD;
        reset_n = 1'b0; // 리셋 활성화 (Active-low)
        $display("[%0t] Reset Asserted.", $time);
        #(CLK_PERIOD * 2);

        reset_n = 1'b1; // 리셋 비활성화
        $display("[%0t] Reset De-asserted. System is ready.", $time);
        #CLK_PERIOD;

        // --- 2. 입력 데이터 생성 및 주입 ---
        generate_input_fmap();
        $display("[%0t] Input feature map generated. Applying to DUT...", $time);
        
        i_in_valid = 1'b1;
        #CLK_PERIOD;
        i_in_valid = 1'b0; // 1-cycle pulse

        // --- 3. 결과 대기 및 확인 ---
        // Total Latency = max_pool(1) + matrixmultiplex(2) = 3 cycles
        // Wait until output is valid
        wait (o_ot_valid);
        $display("[%0t] Output is valid!", $time);
        
        // --- 4. 최종 출력 값 표시 ---
        display_results();

        #(CLK_PERIOD * 5);
        $display("[%0t] Simulation Finished.", $time);
        $finish;
    end
    
    // --- Task: 입력 데이터 생성 ---
    task generate_input_fmap;
        integer i, ch, row, col, pixel_val;
        begin
            pixel_val = 0;
            for (ch = 0; ch < `CI; ch = ch + 1) begin
                for (row = 0; row < `POOL_IN_SIZE; row = row + 1) begin
                    for (col = 0; col < `POOL_IN_SIZE; col = col + 1) begin
                        // 32-bit signed integer로 값 할당
                        i = (ch * `POOL_IN_SIZE * `POOL_IN_SIZE) + (row * `POOL_IN_SIZE) + col;
                        i_in_fmap[(i * `IF_BW) +: `IF_BW] = $signed(pixel_val);
                        pixel_val = pixel_val + 1;
                    end
                end
            end
        end
    endtask

    // --- Task: 결과 출력 ---
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