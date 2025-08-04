class scoreboard extends uvm_scoreboard;
  `uvm_component_utils(scoreboard)
  
  uvm_tlm_analysis_fifo#(apb_packet) input_packet_collected;
  
  apb_packet input_packet;
  
  virtual cnn_intf     cnn_vif;
  virtual apb_if       apb_vif;         
  
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
      data_check();
    join_none
  
  endtask: run_phase
  
  virtual task packet_get();
    forever begin
      input_packet_collected.get(input_packet); 
      packet_compare();
    end
  endtask: packet_get
  
virtual task data_check();
  logic [7:0] WEIGHT;
  logic [7:0] FMAP;
  logic [15:0] EXP_CNN_OUT;

  forever begin
    @(posedge cnn_vif.clk);
    
    if(apb_vif.PSEL && apb_vif.PENABLE && !apb_vif.PWRITE) begin
      case (apb_vif.PADDR)
        32'h00: begin
          @(posedge apb_vif.PCLK);
          WEIGHT = apb_vif.PRDATA;
          `uvm_info(get_type_name(), $sformatf("Weight set to: 0x%02h", WEIGHT), UVM_LOW)
        end

        32'h08: begin
          @(posedge cnn_vif.clk);
          @(posedge apb_vif.PCLK);
          FMAP = apb_vif.PRDATA;
          `uvm_info(get_type_name(), $sformatf("Fmap set to: 0x%02h", FMAP), UVM_LOW)
        end

        32'h04: begin
          // valid read로 간주될 때 비교
          EXP_CNN_OUT = WEIGHT * FMAP;

          repeat(2) @(posedge cnn_vif.clk); // 연산 latency 보정

          if (cnn_vif.cnn_out === EXP_CNN_OUT) begin
            `uvm_info(get_type_name(),
              $sformatf("[PASS] fmap (0x%02h) × weight (0x%02h) = 0x%04h (cnn_out)",
                        FMAP, WEIGHT, EXP_CNN_OUT),
              UVM_LOW)
          end else begin
            `uvm_error(get_type_name(),
              $sformatf("[FAIL] fmap (0x%02h) × weight (0x%02h) = 0x%04h, but cnn_out = 0x%04h",
                        FMAP, WEIGHT, EXP_CNN_OUT, cnn_vif.cnn_out))
          end
        end
      endcase
    end
  end
endtask

  
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
        
  
      
endclass: scoreboard
  