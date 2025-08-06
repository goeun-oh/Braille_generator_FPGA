module valid_gen(
    input clk,
    input reset_n,
    input i_valid,
    output o_valid
);
    reg [$clog2(1000)-1:0]cnt;
    reg r_valid;

    assign o_valid = r_valid;

    always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        cnt <= 0;
        r_valid <= 0;
    end else begin
        if (i_valid && !r_valid) begin
            r_valid <= 1;
            cnt <= 1;
        end else if (r_valid) begin
            if (cnt == 1000 - 1) begin
                r_valid <= 0;
                cnt <= 0;
            end else begin
                cnt <= cnt + 1;
            end
        end
    end
end
endmodule
