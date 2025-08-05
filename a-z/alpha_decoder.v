`include "defines_cnn_core.v"

module alpha_decoder (
    input                         i_valid,
    input  [$clog2(`core_CO)-1:0] index_info,
    output [                 7:0] o_alpha,
    output                        o_valid
);

    reg [7:0] alpha;
    assign o_alpha = alpha;
    assign o_valid = i_valid;

    always @(*) begin
        case (index_info)
            5'd0: begin
                alpha = 8'h61;
            end
            5'd1: begin
                alpha = 8'h62;
            end
            5'd2: begin
                alpha = 8'h63;
            end
            5'd3: begin
                alpha = 8'h64;
            end
            5'd4: begin
                alpha = 8'h65;
            end
            5'd5: begin
                alpha = 8'h66;
            end
            5'd6: begin
                alpha = 8'h67;
            end
            5'd7: begin
                alpha = 8'h68;
            end
            5'd8: begin
                alpha = 8'h69;
            end
            5'd9: begin
                alpha = 8'h6A;
            end
            5'd10: begin
                alpha = 8'h6B;
            end
            5'd11: begin
                alpha = 8'h6C;
            end
            5'd12: begin
                alpha = 8'h6D;
            end
            5'd13: begin
                alpha = 8'h6E;
            end
            5'd14: begin
                alpha = 8'h6F;
            end
            5'd15: begin
                alpha = 8'h70;
            end
            5'd16: begin
                alpha = 8'h71;
            end
            5'd17: begin
                alpha = 8'h72;
            end
            5'd18: begin
                alpha = 8'h73;
            end
            5'd19: begin
                alpha = 8'h74;
            end
            5'd20: begin
                alpha = 8'h75;
            end
            5'd21: begin
                alpha = 8'h76;
            end
            5'd22: begin
                alpha = 8'h77;
            end
            5'd23: begin
                alpha = 8'h78;
            end
            5'd24: begin
                alpha = 8'h79;
            end
            5'd25: begin
                alpha = 8'h7A;
            end
            default: begin
                alpha = 8'h61;
            end
        endcase
    end

endmodule
