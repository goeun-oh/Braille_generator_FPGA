//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/20 18:53:21
// Design Name: 
// Module Name: rom
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

`include "timescale.vh"
`include "defines_cnn_core.vh"

//======================================================================
// FC Layer 가중치(Weight)를 저장하는 ROM (144개 x 7비트)
//======================================================================
module fc_weight_rom (
    input  wire [`FC_WEIGHT_ADDR_W-1:0] addr,
    output reg signed [`W_BW-1:0]       dout
);
    always @(*) begin
        case(addr)
            // Class 0 Weights (48개)
            8'd0:   dout = 7'sd5;
            8'd1:   dout = 7'sd-11;
            8'd2:   dout = 7'sd-15;
            // ... (중략) ...
            8'd47:  dout = 7'sd12;

            // Class 1 Weights (48개)
            8'd48:  dout = 7'sd-20;
            8'd49:  dout = 7'sd17;
            // ... (중략) ...
            8'd95:  dout = 7'sd-14;

            // Class 2 Weights (48개)
            8'd96:  dout = 7'sd20;
            8'd97:  dout = 7'sd-3;
            // ... (중략) ...
            8'd143: dout = 7'sd16;
            
            default: dout = 7'sd0;
        endcase
    end
endmodule


//======================================================================
// FC Layer 편향(Bias)을 저장하는 ROM (3개 x 6비트)
//======================================================================
module fc_bias_rom (
    input  wire [`FC_BIAS_ADDR_W-1:0] addr,
    output reg signed [`BIAS_BW-1:0]  dout
);
    always @(*) begin
        case(addr)
            2'd0: dout = 6'sd11;
            2'd1: dout = 6'sd-7;
            2'd2: dout = 6'sd-15;
            default: dout = 6'sd0;
        endcase
    end
endmodule
