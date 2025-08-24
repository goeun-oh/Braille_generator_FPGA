cnn_intf cnn_vif();

assign cnn_vif.clk          = dut.PCLK;
assign cnn_vif.rst_n        = dut.PRESETN;
assign cnn_vif.cnn_in       = dut.I_IN_FMAP;
assign cnn_vif.cnn_out      = dut.O_OT_KERNEL_ACC;
assign cnn_vif.cnn_valid    = dut.my_ip_0.i_in_valid;
assign cnn_vif.cnn_weight   = dut.my_ip_0.i_cnn_weight;
