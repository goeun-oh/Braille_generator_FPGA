`timescale 1ns / 1ps
// lvl 추가
module color_comparison (
    input  logic       pclk,
    input  logic       reset,
    input  logic       color_done,      // 색 인식 완료 // high면 비교 시작
    // input  logic [1:0] lvl,             // 난이도 (0~3): 비교 횟수 = lvl + 1
    input  logic [7:0] question,        // 정답 색상들 (2bit * 4개)
    input  logic [7:0] detected_color,  // 감지된 색상들 (2bit * 4개)

    // GUI side
    output logic [1:0] crct_incrct,      // win:01 / lose:10
    output logic comparison_done,
    input  logic give_comparison_done,
    output logic [2:0] led_comp
    //added signal
);

    logic [2:0] led_comp_next;
    logic comp_count, comp_count_next;


    logic [1:0] state, state_next;
    logic [1:0] crct_incrct_reg, crct_incrct_next;

    logic comparison_done_next;

    assign crct_incrct = crct_incrct_reg;

    localparam IDLE = 0, CRCT_INCRCT = 1, WAIT = 2;


    
    always_ff @(posedge pclk, posedge reset) begin
        if (reset) begin
            state           <= IDLE;
            crct_incrct_reg <= 2'b0;
            comparison_done <= 0;
            led_comp        <= 0;
            comp_count      <= 0;
        end else begin
            state           <= state_next;
            crct_incrct_reg <= crct_incrct_next;
            comparison_done <= comparison_done_next;
            led_comp        <= led_comp_next;
            comp_count <= comp_count_next;
        end
    end



    // win 01, lose 10
    always_comb begin
        state_next       = state;
        crct_incrct_next = crct_incrct_reg;
        comparison_done_next = comparison_done;
        led_comp_next    = led_comp;
        comp_count_next  = comp_count;
        case (state)

            IDLE: begin
                // state_next 안해줘도 clk 마다 refresh 되서 필요x
                led_comp_next[1:0] = 2'b01;
                state_next = IDLE;
                comparison_done_next = 0;
                crct_incrct_next = 2'b00;
                if (color_done) begin
                    state_next = CRCT_INCRCT;
                end
            end

            CRCT_INCRCT: begin
                led_comp_next[1:0] = 2'b11;
                state_next = WAIT;
                comparison_done_next = 1;
                if (question == detected_color) begin
                    crct_incrct_next = 2'b01;  // win                     
                end else begin
                    crct_incrct_next = 2'b10;  // lose               
                end        
            end

            WAIT : begin
                led_comp_next[1:0] = 2'b10;
                led_comp_next[2] = (comp_count) ? 1:0;
                if(give_comparison_done) begin
                    state_next = IDLE;   
                    comparison_done_next = 0;
                    comp_count_next = comp_count + 1;
                end
            end

        endcase
    end
endmodule