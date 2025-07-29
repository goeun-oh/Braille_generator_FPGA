`timescale 1ns / 1ps

module cnn_top #(
    parameter I_F_BW       = 8,
    parameter O_F_BW       = 20,
    parameter KX           = 5,
    parameter KY           = 5,
    parameter W_BW         = 8,
    parameter CI           = 1,
    parameter CO           = 3,
    parameter IX           = 28,
    parameter IY           = 28,
    parameter B_BW         = 16,
    parameter AK_BW        = 21,
    parameter ACI_BW       = 21,
    parameter AB_BW        = 21,
    parameter AR_BW        = 20,


    parameter BITSHIFT_BW  = 20,

    parameter OUT_W        = IX - KX + 1,
    parameter OUT_H        = IY - KY + 1,
    //pooling//
    parameter ST2_Pool_CI  = 3,
    parameter ST2_Pool_CO  = 3,
    parameter ST2_Conv_CI  = 3,
    parameter ST2_Conv_CO  = 3,
    
    parameter ST2_Conv_IBW = 20,
    parameter ST2_O_F_BW   = 35,
    parameter POOL_OUT_W = 12,
    parameter POOL_OUT_H = 12
) (
    input                                         clk,
    input                                         reset_n,
    input                                         i_valid,
    input [3:0] sw,
    // output                                        w_stage2_core_valid,
    // output [ST2_Conv_CO * (ST2_O_F_BW-1)-1 : 0]   w_stage2_core_fmap,
    // output o_core_valid,
    //output [CO*O_F_BW-1:0] o_core_fmap,
    // output o_core_done
    output out_valid,
    output [7:0] alpha,
    output [2:0] led
);
    wire signed [CO*O_F_BW-1:0] w_core_fmap;
    wire w_core_valid;
    assign o_core_valid = w_core_valid;
    assign o_core_fmap = w_core_fmap;

    wire                                  w_pooling_core_valid;
    wire [ST2_Pool_CI * ST2_Conv_IBW-1:0] w_pooling_core_fmap;

    parameter LATENCY = 1;
    // ===============================
    // cnn_core instance
    // ===============================
    reg signed [         CO*B_BW-1 : 0] w_cnn_bias;
    reg                                 o_done;
    wire       [            I_F_BW-1:0] w_pixel;
    reg signed [CO*CI*KX*KY*W_BW-1 : 0] w_cnn_weight;
    reg signed [                   7:0] rom          [  0:74];
    reg signed [              B_BW-1:0] bias_mem     [0:CO-1];

    // 동은형 출력단
    wire                                        w_stage2_core_valid;
    wire [ST2_Conv_CO * (ST2_O_F_BW-1)-1 : 0]   w_stage2_core_fmap;
    ////////
    integer                             i;


    initial begin
        $readmemh("conv1_weights.mem", rom);
        $readmemh("bias.mem", bias_mem);

    end  
    always @(*) begin
        for (i = 0; i < 75; i = i + 1) begin
            w_cnn_weight[i*W_BW+:W_BW] = rom[i];
        end
        for (i = 0; i < CO; i = i + 1) begin
            w_cnn_bias[i*B_BW+:B_BW] = bias_mem[i];
        end            
    end
    

    wire o_valid;
    fmap_feeder feeder (
        .clk        (clk),
        .reset_n    (reset_n),
        .i_valid    (i_valid),  // 1클럭만 high!
        .sw    (sw),  // 1클럭만 high!
        .o_pixel    (w_pixel),
        .o_out_valid(o_valid)
    );


    cnn_core #(
        .I_F_BW(I_F_BW),
        .KX(KX),
        .KY(KY),
        .W_BW(W_BW),
        .B_BW(B_BW),
        .CI(CI),
        .CO(CO),  // CO=1, 각 core는 1개의 출력 채널만 처리
        .AK_BW(AK_BW),
        .ACI_BW(ACI_BW),
        .O_F_BW(O_F_BW),
        .AB_BW(AB_BW),
        .AR_BW(AR_BW)
    ) u_cnn_core (
        .clk(clk),
        .reset_n(reset_n),
        .i_cnn_weight(w_cnn_weight),
        .i_cnn_bias(w_cnn_bias),
        .i_in_valid(o_valid),
        .i_in_fmap(w_pixel),
        .o_ot_valid(w_core_valid),
        .o_ot_fmap(w_core_fmap)
    );



    wire signed [CO*BITSHIFT_BW-1:0] w_bs_core_fmap;


genvar t;
generate
    for (t = 0; t < CO; t = t + 1) begin : GEN_SHIFT
        wire signed [O_F_BW-1:0] fmap_in  = w_core_fmap[t*O_F_BW +: O_F_BW];
        wire signed [BITSHIFT_BW-1:0] fmap_out = fmap_in >>> (O_F_BW-BITSHIFT_BW);

        assign w_bs_core_fmap[t*BITSHIFT_BW +: BITSHIFT_BW] = fmap_out;
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

    stage3_top_cnn U_stage3_top_cnn(
        .clk(clk),
        .reset_n(reset_n),
        .i_Relu_valid(w_stage2_core_valid),
        .i_in_Relu(w_stage2_core_fmap),
        .o_valid(out_valid),
        .alpha(alpha),
        .led(led)
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
            if (x_cnt == OUT_W - 1) begin
                x_cnt <= 0;
                if (y_cnt == OUT_H - 1) begin
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
        end else if (x_cnt == OUT_W-1 && y_cnt == OUT_H -1) begin
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
            if (x_pool_cnt == POOL_OUT_W - 1) begin
                x_pool_cnt <= 0;
                if (y_cnt == POOL_OUT_H - 1) begin
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
    reg [O_F_BW-1:0] result_fmap[0:CO-1][0:OUT_H-1][0:OUT_W-1];
    reg [ST2_Conv_IBW-1:0] result_pooling_fmap[0:CO-1][0:POOL_OUT_H-1][0:POOL_OUT_W-1];

    integer ch;
    always @(*) begin
        if (w_core_valid) begin
            for (ch = 0; ch < CO; ch = ch + 1) begin
                result_fmap[ch][y_cnt][x_cnt] <= w_core_fmap[ch*O_F_BW+:O_F_BW];
            end
        end
    end
    always @(posedge clk) begin
        if (w_pooling_core_valid) begin
            for (ch = 0; ch < CO; ch = ch + 1) begin
                result_pooling_fmap[ch][y_pool_cnt][x_pool_cnt] <= w_pooling_core_fmap[ch*ST2_Conv_IBW+:ST2_Conv_IBW];
            end
        end
    end

    integer j;
    integer k;
    reg [W_BW-1:0] reg_weight [0:CO-1][0:KY-1][0:KX-1];
    always @(posedge clk) begin
        for (ch=0; ch<CO; ch=ch+1)begin
            for (k= 0; k < KY; k = k + 1) begin
                for (j= 0; j < KX; j = j + 1) begin
                    reg_weight[ch][k][j] <= w_cnn_weight[(ch*KX*KY+k*KY+j)*W_BW +: W_BW];
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