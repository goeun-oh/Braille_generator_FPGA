cnn_intf cnn_vif();

assign cnn_vif.clk          = dut.PCLK;
assign cnn_vif.rst_n        = dut.PRESETN;
assign cnn_vif.cnn_out      = dut.O_OT_KERNEL_ACC;