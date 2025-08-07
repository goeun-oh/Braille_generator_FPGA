class base_test extends uvm_test;
  
  tb_env tb;
  
  `uvm_component_utils(base_test)
  
  function new (string name, uvm_component parent);
    super.new(name , parent);
  endfunction: new
  
  virtual function void build_phase (uvm_phase phase);
    super.build_phase(phase);    
    tb = tb_env::type_id::create("tb", this);
    `uvm_info(get_type_name(), {"build_phase(test_lib)", get_full_name()}, UVM_LOW)
  endfunction: build_phase
  
  virtual task run_phase(uvm_phase phase);
    phase.phase_done.set_drain_time(this, 2000ns);
  endtask: run_phase
  
endclass: base_test