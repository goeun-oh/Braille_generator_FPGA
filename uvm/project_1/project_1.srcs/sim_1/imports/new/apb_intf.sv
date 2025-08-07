interface apb_if;
  logic        PCLK;
  logic        PRESETN;
  
  logic [31:0] PADDR;
  logic        PSEL;
  logic        PENABLE;
  logic        PWRITE;
  logic [31:0] PWDATA;
  logic        PREADY;
  logic [31:0] PRDATA;
endinterface: apb_if