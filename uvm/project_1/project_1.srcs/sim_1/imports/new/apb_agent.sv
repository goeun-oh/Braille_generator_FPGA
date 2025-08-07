class apb_agent extends uvm_agent;
  
  protected uvm_active_passive_enum is_active = UVM_ACTIVE;
  
  apb_driver    driver;
  apb_sequencer sequencer;
  apb_monitor   monitor;
  
  `uvm_component_utils_begin(apb_agent)
  `uvm_field_enum(uvm_active_passive_enum, is_active, UVM_ALL_ON)
  `uvm_component_utils_end
  
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction: new
  
  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);         
    monitor = apb_monitor::type_id::create("monitor", this);
    if(is_active == UVM_ACTIVE) begin
      driver    = apb_driver::type_id::create("driver", this);
      sequencer = apb_sequencer::type_id::create("sequencer", this); 
    end
    `uvm_info(get_type_name(), $sformatf("[BUILD PHASE] APB AGENT %0d", is_active), UVM_LOW)
  endfunction: build_phase
  
  virtual function void connect_phase(uvm_phase phase);
    if(is_active == UVM_ACTIVE) driver.seq_item_port.connect(sequencer.seq_item_export);
    `uvm_info(get_type_name(), "[CONNECT PHASE] connect start", UVM_LOW)
  endfunction: connect_phase
  
endclass: apb_agent