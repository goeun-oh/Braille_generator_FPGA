`include "uvm_macros.svh"
import uvm_pkg::*;
class apb_packet extends uvm_sequence_item;
  rand logic [31:0] ADDR;
  rand logic [31:0] DATA;
  rand logic        WRITE;
   
  `uvm_object_utils_begin(apb_packet)
     `uvm_field_int(ADDR, UVM_ALL_ON)
     `uvm_field_int(DATA, UVM_ALL_ON)
     `uvm_field_int(WRITE, UVM_ALL_ON)
  `uvm_object_utils_end
  
  function new (string name = "apb_packet");
    super.new(name);
  endfunction: new
  
  virtual task packet_display();
    `uvm_info(get_type_name(), $sformatf("ADDR: 0x%03h, DATA: 0x%03h, WRITE: 0x%03h", ADDR, DATA, WRITE), UVM_LOW)
  endtask: packet_display
  
endclass: apb_packet