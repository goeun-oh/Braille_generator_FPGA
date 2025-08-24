interface cnn_intf;
  logic                         clk;
  logic                         rst_n;
  logic [`KX*`KY*`I_F_BW-1 : 0] cnn_in;
  logic [`AK_BW-1 : 0]          cnn_out;
  logic                         cnn_valid;
  logic [`KX*`KY*`W_BW-1 : 0]   cnn_weight;

  logic [`KX*`KY*`I_F_BW-1 : 0] cnn_in_1d;
  logic [`KX*`KY*`I_F_BW-1 : 0] cnn_in_2d;
  logic [`KX*`KY*`I_F_BW-1 : 0] cnn_in_3d;
  logic [`KX*`KY*`W_BW-1 : 0]   cnn_weight_1d;
  logic [`KX*`KY*`W_BW-1 : 0]   cnn_weight_2d;
  logic                         cnn_valid_1d;
  logic                         cnn_valid_2d;
  
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      cnn_in_1d     <= 0;
      cnn_weight_1d <= 0;
      cnn_valid_1d  <= 0;
    end else begin
      cnn_in_1d     <= cnn_in;
      cnn_weight_1d <= cnn_weight;
      cnn_valid_1d  <= cnn_valid;
    end
  end
    
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      cnn_in_2d     <= 0;
      cnn_weight_2d <= 0;
      cnn_valid_2d  <= 0;
    end else begin
      cnn_in_2d     <= cnn_in_1d;
      cnn_weight_2d <= cnn_weight_1d;
      cnn_valid_2d  <= cnn_valid_1d;
    end
  end  
  
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      cnn_in_3d     <= 0;
    end else begin
      cnn_in_3d     <= cnn_in_2d;
    end
  end  
  
endinterface: cnn_intf
