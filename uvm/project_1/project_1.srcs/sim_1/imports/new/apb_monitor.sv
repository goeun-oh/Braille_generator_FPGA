class apb_monitor extends uvm_monitor;
  `uvm_component_utils(apb_monitor)
  
  virtual apb_if apb_vif;
  
  int packet_num;
  
  apb_packet data_collected;
  apb_packet data_clone;
  
  uvm_analysis_port #(apb_packet) item_collected_port;
  
  function new (string name, uvm_component parent);
    super.new(name, parent);
  endfunction: new
  
  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    
    if(!uvm_config_db#(virtual apb_if)::get(this, "", "in_apb_vif", apb_vif))
      `uvm_fatal(get_type_name(), {"virtual interface must be set for: " , get_full_name(), ".vif"})
      
      `uvm_info(get_type_name(), "[BUILD PHASE] apb_monitor build phase", UVM_LOW)
      
      item_collected_port = new("item_collected_port", this);
      data_collected = apb_packet::type_id::create("data_collected");
      data_clone = apb_packet::type_id::create("data_clone");
  endfunction: build_phase
  
  virtual task run_phase(uvm_phase phase);
    `uvm_info(get_type_name(), "run_phase", UVM_LOW)
    fork
       collected_data();
    join_none
  endtask: run_phase
  
  virtual task collected_data();
    `uvm_info(get_full_name( ), "Run stage start.", UVM_LOW)  
    forever begin
      @(negedge apb_vif.PCLK);
      if(apb_vif.PRESETN) begin
        if(apb_vif.PSEL && apb_vif.PENABLE) begin
           data_collected.ADDR  = apb_vif.PADDR;
           data_collected.WRITE = apb_vif.PWRITE;
           if(apb_vif.PWRITE) begin  
             data_collected.DATA = apb_vif.PWDATA;
           end else begin             
             data_collected.DATA = apb_vif.PRDATA;
           end
           $cast(data_clone, data_collected.clone());
           item_collected_port.write(data_clone);  
           packet_num++;  
        end
      end
    end
  endtask: collected_data
      
  virtual function void report_phase(uvm_phase phase);
    `uvm_info(get_type_name(), $sformatf("REPORT: COLLECT PACKET: %0d", packet_num), UVM_LOW)      
  endfunction: report_phase
    
endclass: apb_monitor