class apb_env extends uvm_env;
  
  apb_agent agent;
  
  `uvm_component_utils(apb_env)
  
  function new (string name, uvm_component parent);
    super.new(name, parent);
  endfunction: new
  
  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    uvm_config_db#(int)::set(this, "agent.*", "is_active", UVM_ACTIVE);    
    agent = apb_agent::type_id::create("agent", this);   
    `uvm_info(get_type_name(), "[BUILD PHASE] APB ENV", UVM_LOW)
  endfunction: build_phase
  
endclass: apb_env