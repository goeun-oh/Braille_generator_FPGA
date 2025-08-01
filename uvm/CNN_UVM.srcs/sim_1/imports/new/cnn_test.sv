class cnn_test extends base_test;
  `uvm_component_utils(cnn_test)
  
  cnn_sequence seq;
  
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction: new
  
  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    `uvm_info(get_full_name( ), "build_phase", UVM_LOW)
  endfunction: build_phase
  
  virtual task run_phase(uvm_phase phase);
    `uvm_info(get_full_name( ), "run_phase", UVM_LOW)
    super.run_phase(phase);
    phase.raise_objection(this);
    seq = cnn_sequence::type_id::create("seq");
    seq.start(tb.apb.agent.sequencer);
    phase.drop_objection(this);
  endtask: run_phase
  
endclass: cnn_test