// Register Address encoding
`define REG_WEIGHT      3'd1//8
`define REG_VALID       3'd0//4

// Register Number encoding
`define WEIGHT     2'b10
`define VALID      2'b01

module custom_ip_APB3 (
    input [31:02] PADDR   , // Address (required)
    input         PSEL    , // Slave Select (required)
    input         PENABLE , // Enable (required) (* X_INTERFACE_INFO = "xilinx.com:interface:apb:1.0 S_APB3 PWRITE" *)
    input         PWRITE  , // Write Control (required)
    input [31:0]  PWDATA  , // Write Data (required)
    output        PREADY  , // Slave Ready (required)
    output[31:0]  PRDATA  , // Read Data (required)
    output        PSLVERR , // Slave Error Response (required)
    //  additional ports here
    input         PCLK    ,
    input         PRESETN ,

    output [31:0]  O_CNN_WEIGHT,
    output           O_IN_VALID
  );

// Internal signal decralation
    // AHP porotocol Write & Read Enable
    wire            iwen;
    wire            iren;

    // Write address and write data fetch
    wire [31:02]    paddr;
    wire [31:00]    pwdata;

    // Register Write enable decoding
    wire            cnn_weight_wen;
    wire            in_valid_wen;

   // Register Read enable decoding
    wire            cnn_weight_ren;
    wire            in_valid_ren;

    // Register contents update
    wire [31:0]      next_cnn_weight;
    wire             next_in_valid;


    reg [31:0]     reg_cnn_weight;
    reg [31:0]     reg_in_valid;


    // PRDATA mux for register
    wire [01:00]    isel;
    reg  [31:00]    prdata;

    // error
    wire            error;
  
// Main Code
    // internal read and write enable
    assign iwen = PENABLE & PSEL & PWRITE;
    assign iren = PENABLE & PSEL & ~(PWRITE);
  
     // Write address and write data fetch
    assign paddr  = (PSEL          )? PADDR[31:02]  : 29'd0;
    assign pwdata = (PSEL && PWRITE)? PWDATA : 32'h0000_0000;

    // Register write enable decoding
    assign cnn_weight_wen = ((iwen==1) && (paddr == `REG_WEIGHT))? 1'b1 : 1'b0;
    assign in_valid_wen   = ((iwen==1) && (paddr == `REG_VALID)) ? 1'b1 : 1'b0;

    //write-only Register enable decoding


    // Register Read enable decoding
    assign cnn_weight_ren = ((iren==1) && (paddr == `REG_WEIGHT))? 1'b1 : 1'b0;
    assign in_valid_ren   = ((iren==1) && (paddr == `REG_VALID))? 1'b1 : 1'b0;

    // Read-only Register enable decoding

    // Register contents update
    assign next_cnn_weight = (cnn_weight_wen == 1'b1)? pwdata  : reg_cnn_weight;
    assign next_in_valid   = (in_valid_wen == 1'b1)  ? pwdata[0:0]      : reg_in_valid[0:0];
  
    always @(posedge PCLK, negedge PRESETN) begin
        if(!PRESETN) begin
            reg_cnn_weight   <= 32'd0;
            reg_in_valid     <= 32'd0;
        end
        else begin
            reg_cnn_weight <= next_cnn_weight;
            reg_in_valid   <= {31'd0, next_in_valid};
        end
    end

// Register contents drive
    assign  O_CNN_WEIGHT = reg_cnn_weight;
    assign  O_IN_VALID   = reg_in_valid[0:0];

// Protocol Signal drive
    assign PREADY      = 1'b1;
    assign PSLVERR     = ((PENABLE & PSEL & PREADY) == 1'b1) ? error : 1'b0;

//error
    assign error       = 1'b0;

// PRDATA MUX for register
    assign isel = {cnn_weight_ren, in_valid_ren};
    assign PRDATA = prdata;

  always @(isel, reg_cnn_weight, reg_in_valid) begin
        case(isel)
            `WEIGHT: prdata = reg_cnn_weight;
            `VALID : prdata = reg_in_valid;
            default : prdata = 32'h0000_0000;
        endcase
    end

endmodule
                                       

                                                                                                                                                                               

                                                       
