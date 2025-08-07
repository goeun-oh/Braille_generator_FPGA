`include "defines_cnn_core.v"

module compare_alpha(
    input clk,
    input reset_n,
    input [`ST3_OUT_BW-1:0]i_cnn_value,
    input [$clog2(`core_CO)-1:0] i_index_info,
    input i_valid,
    output o_valid,
    output [$clog2(`core_CO)-1:0] o_index_info
);

    reg signed [`ST3_OUT_BW-1:0] compare_value;
    reg [$clog2(`core_CO)-1:0] compare_index_info;

    reg compare_valid;
    
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            compare_value <=0;
            compare_index_info <= 0;
        end else if (i_valid) begin
            if(!compare_valid) begin
                if($signed(compare_value) <= $signed(i_cnn_value)) begin
                    compare_value <= i_cnn_value;
                    compare_index_info <= i_index_info;
                end
            end 
        end else if (compare_valid)begin
            compare_value <=0;
            compare_index_info <=0;
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            compare_valid <=0;
        end else if(i_index_info == `core_CO -1) begin
            compare_valid <=1;
        end else begin
            compare_valid <=0;
        end
    end

    assign o_index_info = compare_index_info;
    assign o_valid = compare_valid;

endmodule