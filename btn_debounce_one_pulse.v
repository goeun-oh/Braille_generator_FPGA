`timescale 1ns / 1ps

module btn_debounce_one_pulse(
    input clk,
    input reset_n,
    input i_btn,
    output o_btn
    );

    reg [7:0] q_reg, q_next; // shift register
    reg edge_detect;
    wire btn_debounce;

    // 1khz clk
    reg [$clog2(100_000)-1 : 0] counter;
    reg r_1khz;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            counter <= 0;
            r_1khz <= 1'b0;
        end
        else begin
            if(counter == 100_000 - 1) begin
                counter <= 0;
                r_1khz <= 1'b1;
            end
            else begin
                counter <= counter + 1;
                r_1khz <= 1'b0;  // Changed from blocking to non-blocking assignment
            end
        end
    end

    //state logic, shift register
    always@(posedge r_1khz or negedge reset_n) begin
        if (!reset_n) begin
            q_reg <= 0;
        end
        else begin 
            q_reg <= q_next;
        end
    end

    // next logic - FIXED: Added q_reg to sensitivity list
    always @(i_btn, q_reg) begin
        // q_reg 현재의 상위 7bit를 다음 하위 7bit에 넣고, 최상위에는 i_btn 넣기
        q_next = {i_btn, q_reg[7:1]}; // 8 shift의 동작 설명
    end

    // 8 input And gate
    assign btn_debounce = &q_reg;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            edge_detect <= 0;
        end
        else begin
            edge_detect <= btn_debounce;
        end
    end
    
    // 최종 출력
    assign o_btn = btn_debounce & (~edge_detect);

endmodule