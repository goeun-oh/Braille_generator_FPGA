module bram_reader (
    input wire clk,
    input wire reset_n,
    input wire i_btn,
    output reg [ADDR_WIDTH-1:0] bram_addr,
    input wire [DATA_WIDTH-1:0] bram_dout,
    output reg valid,
    output reg [DATA_WIDTH-1:0] data_out
);
    reg [ADDR_WIDTH-1:0] addr_cnt;
    reg w_valid;
    always @(posedge clk or negedge reset_n) begin
        if(reset_n) begin
            w_valid <=0;
        end else if (i_btn) begin
            w_valid <=1;
        end 
    end
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            addr_cnt <= 0;
            bram_addr <= 0;
            data_out <= 0;
            valid <=0;
        end else begin
            // 주소 증가
            if(w_valid) begin
                if (bram_addr == 783) begin
                    bram_addr <=0;
                    valid <=0;
                    addr_cnt <=0;
                    data_out <= 0;
                end else begin
                    bram_addr <= addr_cnt;
                    addr_cnt <= addr_cnt + 1;

                    // 유효한 데이터가 나오기 시작하는 클럭 이후부터 valid 설정
                    valid <= 1;
                    data_out <= bram_dout;  // 읽은 데이터 외부에 출력                    
                end
            end
        end
    end
endmodule