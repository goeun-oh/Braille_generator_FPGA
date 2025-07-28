`timescale 1ns / 1ps

module btn_debounce_one_pulse(
  input        clk,
  input        reset_n,
  input        i_btn,
  output reg   o_btn
);

  // debounce용 shift register
  reg [7:0] q_reg;
  wire      btn_debounce;

  // debounce shift
  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      q_reg <= 8'd0;
    end else begin
      q_reg <= {i_btn, q_reg[7:1]};
    end
  end

  // 8 비트가 모두 1이 되었을 때만 '클린' 신호로 간주
  assign btn_debounce = &q_reg;

  // one pulse 생성용 previous 상태 저장
  reg btn_debounce_d;

  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      btn_debounce_d <= 1'b0;
    end else begin
      btn_debounce_d <= btn_debounce;
    end
  end

  // rising edge(버튼이 처음 눌릴 때)만 1클럭 o_btn 출력
  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      o_btn <= 1'b0;
    end else begin
      o_btn <= btn_debounce & ~btn_debounce_d;
    end
  end

endmodule