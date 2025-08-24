`define KX     2   // Number of Kernel X
`define KY     2   // Number of Kernel Y
`define I_F_BW 8   // Bit Width of Input Feature
`define W_BW   8   // BW of weight parameter
`define AK_BW  21  // M_BW + log(KY*KX) Accum Kernel 
`define M_BW   16  // I_F_BW * W_BW

class scoreboard extends uvm_scoreboard;
  `uvm_component_utils(scoreboard)
  
  uvm_tlm_analysis_fifo#(apb_packet) input_packet_collected;
  
  apb_packet input_packet;
  
  virtual cnn_intf     cnn_vif;
  virtual apb_if       apb_vif;  
  
  //CNN_VALUE
  logic [`KX*`KY*`I_F_BW-1 : 0] frame_in[int][int];
  logic [`KX*`KY*`W_BW-1 : 0]   frame_weight[int][int];
  logic [`M_BW-1:0]             frame_mul[int][int];
  logic [`AK_BW-1 : 0]          exp_out;
  
  function new (string name, uvm_component parent);
    super.new(name, parent);
  endfunction: new
  
  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    input_packet_collected = new("input_packet_collected", this);
    input_packet = apb_packet::type_id::create("input_packet");   
    uvm_config_db#(virtual cnn_intf)::get(this, "", "cnn_vif", cnn_vif);    
    uvm_config_db#(virtual apb_if)::get(this, "", "apb_vif", apb_vif);    
    `uvm_info(get_type_name(), "[BUILD_PHASE] scoreboard build", UVM_LOW)
  endfunction: build_phase
  
  virtual task run_phase(uvm_phase phase);
    super.run_phase(phase);
    `uvm_info(get_type_name(), "[RUN_PHASE] scoreboard run_phase", UVM_LOW)
    
    fork
      packet_get();
      cnn_data_check();
    join_none
  
  endtask: run_phase
  
  virtual task packet_get();
    forever begin
      input_packet_collected.get(input_packet); 
      packet_compare();
    end
  endtask: packet_get
    
  virtual task packet_compare();
    logic [31:0] EXP_ADDR;
    logic [31:0] EXP_DATA;
    logic [31:0] EXP_WRITE;
    
    EXP_ADDR  = input_packet.ADDR;
    EXP_DATA  = input_packet.DATA;
    EXP_WRITE = input_packet.WRITE;
    
    if(EXP_ADDR == apb_vif.PADDR) `uvm_info(get_type_name(), $sformatf("[Packet] ADDR: 0x%03h, [DUT] ADDR: 0x%03h", EXP_ADDR, apb_vif.PADDR), UVM_LOW)
    else `uvm_error(get_type_name(), $sformatf("Packet ADDR: 0x%03h, DUT ADDR: 0x%03h", EXP_ADDR, apb_vif.PADDR))
    
    if(EXP_WRITE) begin
      if(EXP_DATA == apb_vif.PWDATA) `uvm_info(get_type_name(), $sformatf("[Packet] WDATA: 0x%03h, [DUT] WDATA: 0x%03h", EXP_DATA, apb_vif.PWDATA), UVM_LOW)
      else `uvm_error(get_type_name(), $sformatf("Packet ADDR: 0x%03h, DUT ADDR: 0x%03h", EXP_DATA, apb_vif.PWDATA))
    end else begin
      if(EXP_DATA == apb_vif.PRDATA) `uvm_info(get_type_name(), $sformatf("[Packet] RDATA: 0x%03h, [DUT] RDATA: 0x%03h", EXP_DATA, apb_vif.PRDATA), UVM_LOW)
      else `uvm_error(get_type_name(), $sformatf("Packet ADDR: 0x%03h, DUT ADDR: 0x%03h", EXP_DATA, apb_vif.PRDATA))
    end
  endtask: packet_compare
        
  virtual task cnn_data_check();
    forever begin
      @(negedge cnn_vif.clk)
      if(cnn_vif.cnn_valid_2d) begin
        //CNN_MODEL
        for (int k = 0; k < `KY; k = k + 1) begin
          for (int j = 0; j < `KX; j = j + 1) begin
            frame_in[k][j]     = cnn_vif.cnn_in_3d[(k*`KX + j)*`I_F_BW +: `I_F_BW];
            frame_weight[k][j] = cnn_vif.cnn_weight_2d[(k*`KX + j)*`W_BW +: `W_BW];
            frame_mul[k][j]    = frame_in[k][j] * frame_weight[k][j];
          end
        end     
        exp_out = 0;
        for (int k = 0; k < `KY; k = k + 1) begin
          for (int j = 0; j < `KX; j = j + 1) begin
            exp_out = exp_out + frame_mul[k][j];
          end
        end
        //CHECK
        if(exp_out == cnn_vif.cnn_out) begin
          `uvm_info(get_type_name(), $sformatf("[PASS] exp_out = %04h, rtl_out = %04h", exp_out, cnn_vif.cnn_out), UVM_LOW)
        end else begin
          `uvm_error(get_type_name(), $sformatf("[ERROR] exp_out = %04h, rtl_out = %04h", exp_out, cnn_vif.cnn_out))
        end
      end
    end
  endtask: cnn_data_check
      
endclass: scoreboard
  
