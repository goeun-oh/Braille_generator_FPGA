apb_if apb_vif();

assign apb_vif.PCLK    = dut.PCLK;
assign apb_vif.PRESETN = dut.PRESETN;
