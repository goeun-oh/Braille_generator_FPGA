class apb_driver extends uvm_driver #(apb_packet);
  `uvm_component_utils(apb_driver)
  
  virtual apb_if apb_vif;
  
  function new (string name, uvm_component parent);
    super.new(name, parent);
  endfunction: new
  
  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    
    if(!uvm_config_db#(virtual apb_if)::get(this, "", "in_apb_vif", apb_vif)) 
      `uvm_fatal(get_type_name(), {"virtual interface must be set for: ", get_full_name(), ".vif"})
      
      `uvm_info(get_type_name(), "[BUILD PHASE] apb_driver build", UVM_LOW)
  endfunction : build_phase
  
  virtual task run_phase(uvm_phase phase);
    `uvm_info(get_type_name(), "[RUN PHASE] driver run_phase start", UVM_LOW)
    fork
       reset();
       drive();
    join
  endtask: run_phase
    
  virtual task reset();
    forever begin
      @(negedge apb_vif.PRESETN);
      `uvm_info(get_type_name(), "negedge apb_vif.PRESETN", UVM_LOW)
      apb_vif.PADDR   = 0;
      apb_vif.PSEL    = 0;
      apb_vif.PENABLE = 0;
      apb_vif.PWRITE  = 0;
      apb_vif.PWDATA  = 0;
    end
  endtask: reset
    
  virtual task drive();
     `uvm_info(get_type_name( ), "Drive signals ... ", UVM_LOW)
     @(posedge apb_vif.PRESETN);
      forever begin
        while(apb_vif.PRESETN != 1'b0) begin
          seq_item_port.get_next_item(req);
          if(req.WRITE) do_write(req.ADDR, req.DATA);
          else do_read(req.ADDR, req.DATA);
          seq_item_port.item_done();
        end
      end
  endtask: drive
  
  virtual task do_write(input logic [31:0] ADDR, input logic [31:0] DATA);
    @(posedge apb_vif.PCLK);
    apb_vif.PADDR    = ADDR;
    apb_vif.PWRITE   = 1; 
    apb_vif.PSEL     = 1;
    apb_vif.PWDATA   = DATA;
    @(posedge apb_vif.PCLK);
    apb_vif.PENABLE  = 1;
    @(posedge apb_vif.PCLK);
    apb_vif.PSEL     = 0;
    apb_vif.PENABLE  = 0;
    @(posedge apb_vif.PCLK);
  endtask: do_write
    
  virtual task do_read(input logic [31:0] ADDR, output logic [31:0] DATA);
    @(posedge apb_vif.PCLK);
    apb_vif.PADDR   = ADDR;
    apb_vif.PWRITE  = 0;
    apb_vif.PSEL    = 1;
    @(posedge apb_vif.PCLK);
    apb_vif.PENABLE = 1;    
    @(posedge apb_vif.PCLK);
    DATA = apb_vif.PRDATA;
    apb_vif.PSEL    = 0;
    apb_vif.PENABLE = 0;
    @(posedge apb_vif.PCLK);
  endtask: do_read

endclass: apb_driver