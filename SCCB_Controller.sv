`timescale 1ns / 1ps


module SCCB_Controller (
    input  logic       clk,
    input  logic       reset,
    input  logic [7:0] reg_addr,
    input  logic [7:0] rom_data,
    output logic [7:0] rom_addr,
    output logic       scl,
    output logic       sda
);
    typedef enum {
        IDLE,
        START,
        DATA,
        STAY,
        STOP
    } state_e;


    state_e state;
    logic start;
    logic [1:0] tick_cnt;  // 0, 1, 2, 3
    logic [3:0] bit_cnt;  // 0~15
    logic [1:0] cycle_cnt;  // 0, 1, 2
    logic [7:0] rom_addr_cnt;  // 0~74
    logic [1:0] wait_5us;  // 0, 1
    logic [7:0] id;  // 0x42
    logic [6:0] wait_250us;
    logic tick_400kHz;

    assign rom_addr = rom_addr_cnt;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            start <= 1'b1; 
        end else if (tick_400kHz) begin
            start <= 1'b0;
        end
    end
    always_comb begin
        scl = 1'b1;
        case(tick_cnt)
        2'b00 : scl = 1'b0; 
        2'b01 : scl = 1'b1;
        2'b10 : scl = 1'b1;
        2'b11 : scl = 1'b0;
        endcase
    end


    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            state <= IDLE;
            wait_5us <= 0;
            tick_cnt <= 1;
            id <= 8'h42;
            cycle_cnt <= 0;
            sda <= 1'b1;
            bit_cnt <= 0;
            rom_addr_cnt <= 0;
            wait_250us <= 0;
        end else begin
            case (state)
                IDLE: begin
                    rom_addr_cnt <= 0;
                    tick_cnt <= 1;
                    sda <= 1'b1;
                    if (start) begin
                        state <= START;
                    end else begin
                        state <= IDLE;
                    end
                end

                START: begin
                    sda <= 1'b0;
                    if (tick_400kHz) begin
                        if (wait_5us == 1) begin
                            wait_5us <= wait_5us + 1;
                            tick_cnt <= 3;
                        end else if (wait_5us == 2) begin
                            state <= DATA;
                            tick_cnt <= 0;
                            wait_5us <= 0;
                            bit_cnt <= 1;
                        end else begin
                            wait_5us <= wait_5us + 1;
                        end
                    end
                end

                DATA: begin
                    if (tick_400kHz) begin
                        tick_cnt <= tick_cnt + 1;
                        if (tick_cnt == 3) begin
                            if (bit_cnt == 8) begin
                                cycle_cnt <= cycle_cnt + 1;
                                bit_cnt   <= 0;
                                sda = 1'b0;
                            end else if (cycle_cnt == 3) begin
                                state <= STOP;
                            end else begin
                                bit_cnt <= bit_cnt + 1;
                                case (cycle_cnt)
                                    2'b00: sda <= id[7-bit_cnt];
                                    2'b01: sda <= reg_addr[7-bit_cnt];
                                    2'b10: sda <= rom_data[7-bit_cnt];
                                    2'b11: sda <= 1'b0;
                                endcase
                            end
                        end
                    end
                end

                STOP: begin
                    cycle_cnt <= 0;
                    bit_cnt   <= 0;
                    if (tick_400kHz) begin
                        if (wait_5us == 1) begin
                            state <= STAY;
                            wait_5us <= 0;
                            sda <= 1'b0;
                        end else begin
                            state <= STOP;
                            wait_5us <= wait_5us + 1;
                            tick_cnt <= 1;
                            sda <= 1'b0;
                        end
                    end
                end

                STAY: begin
                    sda <= 1'b1;
                    if (rom_addr_cnt == 74) begin
                        state <= IDLE;
                    end else begin
                        if (tick_400kHz) begin
                            if (wait_250us == 99) begin
                                state <= START;
                                rom_addr_cnt <= rom_addr_cnt + 1;
                                wait_250us <= 0;
                            end else begin
                                state <= STAY;
                                wait_250us <= wait_250us + 1;
                            end
                        end
                    end
                end
            endcase
        end
    end

    tick_400kHz U_tick_400KHz (.*);
endmodule
