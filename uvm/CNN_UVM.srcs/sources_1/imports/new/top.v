`include "apb3_ip.v"
`include "cnn_kernel.v"

// Code your design here
module top (
        (* X_INTERFACE_INFO = "xilinx.com:interface:apb:1.0 S_APB3 PADDR" *)
    input [31:0]  PADDR   , // Address (required)
    (* X_INTERFACE_INFO = "xilinx.com:interface:apb:1.0 S_APB3 PSEL" *)
    input         PSEL    , // Slave Select (required)
    (* X_INTERFACE_INFO = "xilinx.com:interface:apb:1.0 S_APB3 PENABLE" *)
    input         PENABLE , // Enable (required)
    (* X_INTERFACE_INFO = "xilinx.com:interface:apb:1.0 S_APB3 PWRITE" *)
    input         PWRITE  , // Write Control (required)
    (* X_INTERFACE_INFO = "xilinx.com:interface:apb:1.0 S_APB3 PWDATA" *)
    input [31:0]  PWDATA  , // Write Data (required)
    (* X_INTERFACE_INFO = "xilinx.com:interface:apb:1.0 S_APB3 PREADY" *)
    output        PREADY  , // Slave Ready (required)
    (* X_INTERFACE_INFO = "xilinx.com:interface:apb:1.0 S_APB3 PRDATA" *)
    output [31:0] PRDATA  , // Read Data (required)
    (* X_INTERFACE_INFO = "xilinx.com:interface:apb:1.0 S_APB3 PSLVERR" *)
    output        PSLVERR // Slave Error Response (required)
    //  additional ports here
    , input         PCLK
    , input         PRESETN
    , output wire [15:0] O_OT_KERNEL_ACC
);
        wire    [8-1:0]   O_CNN_WEIGHT;
        wire    [8-1:0]   O_IN_FMAP;
        wire              O_IN_VALID;

   custom_ip_APB3 apb3_reg_0(
           .PADDR        (PADDR[31:02] ),
           .PSEL         (PSEL         ),
           .PENABLE      (PENABLE      ),
           .PWRITE       (PWRITE       ),
           .PWDATA       (PWDATA       ),
           .PREADY       (PREADY       ),
           .PRDATA       (PRDATA       ),
           .PSLVERR      (PSLVERR      ),
           .PCLK         (PCLK         ),
           .PRESETN      (PRESETN      ),
           .O_CNN_WEIGHT (O_CNN_WEIGHT ),
           .O_IN_FMAP    (O_IN_FMAP    ),
           .O_IN_VALID   (O_IN_VALID   )
   );

    cnn_kernel my_ip_0(
          .clk             (PCLK            ),
          .reset_n         (PRESETN         ),
          .i_cnn_weight    (O_CNN_WEIGHT    ),
          .i_in_valid      (O_IN_VALID      ),
          .i_in_fmap       (O_IN_FMAP       ),
          .o_ot_valid      ( ),
          .o_ot_kernel_acc (O_OT_KERNEL_ACC )
    );
    endmodule                                
//apb vip 를 만들어서 검증하려한다. (vip 용어: 이미 검증된 uvm 코드)
//apb protocol 에 맞춘 driver를 만들었따~~
