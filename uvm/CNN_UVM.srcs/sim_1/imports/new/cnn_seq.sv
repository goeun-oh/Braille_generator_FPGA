class cnn_sequence extends uvm_sequence #(apb_packet);
  `uvm_object_utils(cnn_sequence)
  
  function new (string name = "cnn_sequence");
    super.new(name);
  endfunction: new
  
  virtual task HW_WRITE(input logic [31:0] addr, input logic [31:0] wdata);
    `uvm_info(get_type_name(), $sformatf("[HW_WRITE] ADDR: 0x%03h, WDATA: 0x%03h", addr, wdata), UVM_LOW)
    
    `uvm_create(req)
    req.ADDR  = addr;
    req.DATA  = wdata;
    req.WRITE = 1;
    `uvm_send(req)
  endtask : HW_WRITE
  
  virtual task HW_READ(input logic [31:0] addr, output logic [31:0] rdata);  
    `uvm_create(req)
    req.ADDR = addr;
    req.DATA = 0;
    req.WRITE = 0;
    `uvm_send(req)
    rdata = req.DATA;
    
    `uvm_info(get_type_name(), $sformatf("[HW_READ] ADDR: 0x%03h, RDATA: 0x%03h", addr, rdata), UVM_LOW)
  endtask: HW_READ
  
  virtual task ONE_RANDOM_DATA();
    `uvm_do(req)
    req.packet_display();
  endtask: ONE_RANDOM_DATA
  
  virtual task RANDOM_DATA();
    repeat(16) begin
      `uvm_do(req)
      req.packet_display();
    end
  endtask: RANDOM_DATA
        
  virtual task body();
    int rdata;
    
    `uvm_info(get_type_name(), "cnn_seq body() start", UVM_LOW)
    
    HW_WRITE(32'h0008, 32'h0004);
    HW_WRITE(32'h0004, 32'h0001);
    HW_WRITE(32'h0000, 32'h0002);
    
    HW_READ(31'h0008, rdata);
    HW_READ(31'h0004, rdata);
    HW_READ(31'h0000, rdata);
  endtask: body
  
endclass: cnn_sequence