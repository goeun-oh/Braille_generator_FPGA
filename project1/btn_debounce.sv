`timescale 1ns / 1ps

module btn_debounce(
    input clk,
    input reset,
    input i_btn,
    output o_btn
    );


    //reg state, next;
    reg [7:0] q_reg, q_next; // shift register
    reg edge_detect;
    wire btn_debounce;

    // 1khz clk  -> 100_000
    reg [$clog2(10)-1 : 0] counter;
    reg r_1khz;

    always @(posedge clk, posedge reset) begin
        if (reset) begin
            counter <= 0;
        end

        else begin
            if(counter == 10 -1) begin
                r_1khz <=0; // 8888이 뜨지 않게 하기 위해 
                counter <= 0;
                r_1khz <= 1'b1;
            end

            else begin
                counter <= counter + 1;
                r_1khz = 1'b0;
            end
        end

    end

    //state logic, shift register
    always@(posedge r_1khz, posedge reset) begin
        if (reset) begin
            q_reg <= 0;
        end

        else begin 
            q_reg <= q_next;
        end
    end


    // next logic
    always @(i_btn, r_1khz) begin
        // q_reg 현재의 상위 7bit를 다음 하위 7bit에 넣고, 최상위에는 i_btn 넣기
        q_next = {i_btn, q_reg[7:1]}; // 8 shift의 동작 설명
    end

    // 8 input And gate
    assign btn_debounce = &q_reg;

    always @(posedge clk, posedge reset) begin
        if (reset) begin
            edge_detect <= 0;
        end

        else begin
            edge_detect <= btn_debounce;
        end

    end
    
    // 최종 출력
    assign o_btn = btn_debounce & (~edge_detect);

endmodule
