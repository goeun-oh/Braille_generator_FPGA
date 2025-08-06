`timescale 1ns / 1ps
`include "defines_cnn_core.v"
module cnn_top (
    input       clk,
    input       reset_n,
    input       i_valid,
    input [31:0] i_pixel,
    input [3:0] sw,
    // output                                        w_stage2_core_valid,
    // output [`ST2_Conv_CO * (`ST2_O_F_BW-1)-1 : 0]   w_stage2_core_fmap,
    // output o_core_valid,
    //output [CO*O_F_BW-1:0] o_core_fmap,
    // output o_core_done
    output       out_valid,
    output [7:0] alpha,
    output [3:0] led,
    output led_r,
    output led_g,
    output led_b
);
    wire signed [`ST1_CO*`ST1_O_F_BW-1:0] w_core_fmap;
    wire w_core_valid;
    assign o_core_valid = w_core_valid;
    assign o_core_fmap  = w_core_fmap;

    wire                                    w_pooling_core_valid;
    wire [`ST2_Pool_CI * `ST2_Conv_IBW-1:0] w_pooling_core_fmap;

    parameter LATENCY = 1;
    // ===============================
    // cnn_core instance
    // ===============================
    reg signed [`ST1_CO*`ST1_B_BW-1 : 0] w_cnn_bias;
    reg                          o_done;
    wire       [`ISP_BW-1:0]          w_pixel;
    reg signed [`ST1_CO*`ST1_CI*`KX*`KY*`ST1_W_BW-1 : 0] w_cnn_weight;
    reg signed [7:0] rom [0:74];
    reg signed [`ST1_B_BW-1:0] bias_mem [0:`ST1_CO-1];

    // 동은형 출력단
    wire                                          w_stage2_core_valid;
    wire [(`ST2_Conv_CO * (`ST2_O_F_BW))-1 : 0]   w_stage2_core_fmap;
    ////////
    integer i;


    initial begin
        $readmemh("conv1_weights.mem", rom);
        $readmemh("conv1_bias.mem", bias_mem);

    end  
    always @(*) begin
        for (i = 0; i < 75; i = i + 1) begin
            w_cnn_weight[i*`ST1_W_BW+:`ST1_W_BW] = rom[i];
        end
        for (i = 0; i < `ST1_CO; i = i + 1) begin
            w_cnn_bias[i*`ST1_B_BW+:`ST1_B_BW] = bias_mem[i];
        end            
    end
    



    wire [7:0] grayed_px;
    wire grayed_o_valid;    

    gray_filter u_gray_filter(
        .clk(clk),
        .reset_n(reset_n),
        .one_px(w_pixel),           // 32 bit
        .i_in_valid(i_valid),                            
        .grayed_one_px(grayed_px),    // 8 bit
        .o_valid(grayed_o_valid)
    );




    cnn_core u_cnn_core (
        .clk(clk),
        .reset_n(reset_n),
        .i_cnn_weight(w_cnn_weight),
        .i_cnn_bias(w_cnn_bias),
        .i_in_valid(grayed_o_valid),
        .i_in_fmap(grayed_px),
        .o_ot_valid(w_core_valid),
        .o_ot_fmap(w_core_fmap)
    );


    // ===============================
    // bit_shift after Conv1 
    // ===============================
    wire signed [`ST1_CO*(`ST1_O_F_BW-`ST1_BITSHIFT_BW)-1:0] w_bs_core_fmap;


    genvar t;
    generate
        for (t = 0; t < `ST1_CO; t = t + 1) begin : GEN_SHIFT
            wire signed [`ST1_O_F_BW-1:0] bs_temp_in  = $signed(w_core_fmap[t*`ST1_O_F_BW +: `ST1_O_F_BW]);
            wire signed [`ST1_O_F_BW-`ST1_BITSHIFT_BW-1:0] bs_temp_out = bs_temp_in >>> (`ST1_BITSHIFT_BW);

            assign w_bs_core_fmap[t*(`ST1_O_F_BW-`ST1_BITSHIFT_BW) +: (`ST1_O_F_BW-`ST1_BITSHIFT_BW)] = bs_temp_out;
        end
    endgenerate

    // ===============================
    // stage2_pooling instance
    // ===============================
    stage2_pooling_core u_stage2_pooling_core (
        .clk       (clk),
        .reset_n   (reset_n),
        .i_in_valid(w_core_valid),
        // .i_in_fmap (w_core_fmap),
        .i_in_fmap (w_bs_core_fmap),
        .o_ot_valid(w_pooling_core_valid),
        .o_ot_fmap (w_pooling_core_fmap)
    );
    // ===============================
    // stage2_convolution instance
    // ===============================
    stage2_conv u_stge2_conv(
        .clk             (clk),
        .reset_n         (reset_n),
        .i_in_valid      (w_pooling_core_valid),
        .i_in_fmap       (w_pooling_core_fmap),
        .o_ot_valid      (w_stage2_core_valid),
        .o_ot_fmap       (w_stage2_core_fmap)
    );


    // ===============================
    // bit_shift after Conv2 
    // ===============================

    wire signed [`ST2_Conv_CO*(`ST2_O_F_BW-`ST2_BITSHIFT_BW)-1:0] w_bs_stage2_core_fmap;

generate
    for (t = 0; t < `ST2_Conv_CO; t = t + 1) begin : GEN_SHIFT2
        wire signed [`ST2_O_F_BW-1:0] bs2_temp_in  = $signed(w_stage2_core_fmap[t*`ST2_O_F_BW +: `ST2_O_F_BW]);
        wire signed [`ST2_O_F_BW-`ST2_BITSHIFT_BW-1:0] bs2_temp_out = bs2_temp_in >>> (`ST2_BITSHIFT_BW);

        assign w_bs_stage2_core_fmap[t*(`ST2_O_F_BW-`ST2_BITSHIFT_BW) +: (`ST2_O_F_BW-`ST2_BITSHIFT_BW)] = bs2_temp_out;
    end
endgenerate    

    // ===============================
    // stage3_convolution instance
    // ===============================

    stage3_top_cnn U_stage3_top_cnn(
        .clk(clk),
        .reset_n(reset_n),
        .i_Relu_valid(w_stage2_core_valid),
        .i_in_Relu(w_bs_stage2_core_fmap),
        .o_valid(out_valid),
        .alpha(alpha),
        .led(led),
        .led_r(led_r),
        .led_g(led_g),
        .led_b(led_b)
    );

    // ===============================
    // Output coordinate counters
    // ===============================
    reg [4:0] x_cnt, y_cnt;
    reg core_done;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_cnt <= 0;
            y_cnt <= 0;
        end else if (w_core_valid) begin
            if (x_cnt == `ST1_OUT_W - 1) begin
                x_cnt <= 0;
                if (y_cnt == `ST1_OUT_H - 1) begin
                    y_cnt <= 0;
                end else begin
                    y_cnt <= y_cnt + 1;
                end
            end else begin
                x_cnt <= x_cnt + 1;
            end
        end
    end
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            core_done <=0;
        end else if (x_cnt == `ST1_OUT_W -1 && y_cnt == `ST1_OUT_H -1) begin
            core_done <=1;
        end else begin
            core_done <=0;
        end
    end
    // assign o_core_done = core_done;   
    reg [4:0] x_pool_cnt, y_pool_cnt;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x_pool_cnt <= 0;
            y_pool_cnt <= 0;
        end else if (w_pooling_core_valid) begin
            if (x_pool_cnt == `ST1_POOL_OUT_W - 1) begin
                x_pool_cnt <= 0;
                if (y_cnt == `ST1_POOL_OUT_H - 1) begin
                    y_pool_cnt <= 0;
                end else begin
                    y_pool_cnt <= y_pool_cnt + 1;
                end
            end else begin
                x_pool_cnt <= x_pool_cnt + 1;
            end
        end
    end
    reg [LATENCY-1 : 0] r_valid;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_valid <= {LATENCY{1'b0}};
        end else begin
            r_valid[LATENCY-1] <= w_core_valid;
        end
    end
    // ===============================
    // Output fmap memory: [CO][24][24]
    // ===============================
    reg [`ST1_O_F_BW-1:0] result_fmap[0:`ST1_CO-1][0:`ST1_OUT_H-1][0:`ST1_OUT_W-1];
    reg [`ST2_Conv_IBW-1:0] result_pooling_fmap[0:`ST1_CO-1][0:`ST1_POOL_OUT_H-1][0:`ST1_POOL_OUT_W-1];

    integer ch;
    always @(*) begin
        if (w_core_valid) begin
            for (ch = 0; ch < `ST1_CO; ch = ch + 1) begin
                result_fmap[ch][y_cnt][x_cnt] <= w_core_fmap[ch*`ST1_O_F_BW+:`ST1_O_F_BW];
            end
        end
    end
    always @(posedge clk) begin
        if (w_pooling_core_valid) begin
            for (ch = 0; ch < `ST1_CO; ch = ch + 1) begin
                result_pooling_fmap[ch][y_pool_cnt][x_pool_cnt] <= w_pooling_core_fmap[ch*`ST2_Conv_IBW+:`ST2_Conv_IBW];
            end
        end
    end

    integer j;
    integer k;
    reg [`ST1_W_BW-1:0] reg_weight [0:`ST1_CO-1][0:`KY-1][0:`KX-1];
    always @(posedge clk) begin
        for (ch=0; ch<`ST1_CO; ch=ch+1)begin
            for (k= 0; k < `KY; k = k + 1) begin
                for (j= 0; j < `KX; j = j + 1) begin
                    reg_weight[ch][k][j] <= w_cnn_weight[(ch*`KX*`KY+k*`KY+j)*`ST1_W_BW +: `ST1_W_BW];
                end
            end
        end
    end
    // ===============================
    // Done signal: after last pixel
    // ===============================
    //always @(posedge clk or negedge reset_n) begin
    //    if (!reset_n) begin
    //        o_done <= 0;
    //    end else if (&w_core_valid && (x_cnt == OUT_W-1) && (y_cnt == OUT_H-1)) begin
    //        o_done <= 1;
    //    end
    //end

endmodule
