//----------------------------------------------------------------------
//   Copyright 2013 Verilab, Inc.
//   All Rights Reserved Worldwide
//
//   Licensed under the Apache License, Version 2.0 (the
//   "License"); you may not use this file except in
//   compliance with the License.  You may obtain a copy of
//   the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in
//   writing, software distributed under the License is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//   CONDITIONS OF ANY KIND, either express or implied.  See
//   the License for the specific language governing
//   permissions and limitations under the License.
//----------------------------------------------------------------------
`timescale 1ns/100ps

`include "tb_pkg.sv"
`include "apb_intf.sv"
`include "cnn_intf.sv"
module tb_top;

    import tb_pkg::*;
    import uvm_pkg::*;
 
    logic clk;
    logic rst_n;
     
    // virtual interface 
    `include "apb_intf_inst.sv"
    `include "cnn_intf_inst.sv"
 
    // Target DUT with APB3
    top dut (
      .PADDR   (apb_vif.PADDR    ),
      .PSEL    (apb_vif.PSEL     ),
      .PENABLE (apb_vif.PENABLE  ),
      .PWRITE  (apb_vif.PWRITE   ),
      .PWDATA  (apb_vif.PWDATA   ),
      .PREADY  (apb_vif.PREADY   ),
      .PRDATA  (apb_vif.PRDATA   ),
      .PSLVERR (  ),
      .PCLK    (clk              ),  
      .PRESETN (rst_n            ),   
      .O_OT_KERNEL_ACC (  ) 
    );
    
    // clock generation
    always #5 clk = ~clk;

    initial begin
      clk = 1'b0;
      rst_n = 1'b1;
      #1 rst_n = 1'b0;
      #2 rst_n = 1'b1;
    end
  
    //interface set
    initial begin
      uvm_config_db#(virtual apb_if)::set(uvm_root::get(), "uvm_test_top.*", "in_apb_vif", apb_vif);
      uvm_config_db#(virtual cnn_intf)::set(uvm_root::get(), "uvm_test_top.*", "cnn_vif", cnn_vif);
      uvm_config_db#(virtual apb_if)::set(uvm_root::get(), "uvm_test_top.*", "apb_vif", apb_vif);
    end
   
   //test start
    initial begin                   
       run_test("cnn_test");
    end
  
	initial begin
    	$dumpfile("dump.vcd");
      	$dumpvars();
    	#10000ns $finish;
    end
endmodule