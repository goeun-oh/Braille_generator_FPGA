class tb_env extends uvm_env;
  
  scoreboard        sb;
  apb_env           apb;
  
  `uvm_component_utils(tb_env)
  
  function new (string name, uvm_component parent = null);
    super.new(name, parent);
  endfunction: new
  
  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);  
    sb = scoreboard::type_id::create("sb", this);
    apb = apb_env::type_id::create("apb", this);
    `uvm_info(get_full_name( ), "build_phase", UVM_LOW)
  endfunction: build_phase
  
  virtual function void connect_phase(uvm_phase phase);
    apb.agent.monitor.item_collected_port.connect(sb.input_packet_collected.analysis_export);
   `uvm_info(get_full_name( ), "Connect phase complete.", UVM_LOW)
  endfunction: connect_phase
  
endclass: tb_env