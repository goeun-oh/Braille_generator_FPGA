`timescale 1ns / 1ps
// top module

module OV7670_VGA_Display (
    output logic [3:0] test_led,
    input  logic       clk_25MHz,
    // global signals
    input  logic       clk,
    input  logic       reset,
    // control signals
    input  logic [1:0] lvl,
    input  logic       q_start,
    input  logic [7:0] question,
    input  logic       give_done,
    // switch signals
    input  logic       upscale,
    input  logic       mode_chroma,
    // input  logic       median,
    // ov7670 signals
    input  logic       btn_sccb,
    output logic       ov7670_xclk,
    input  logic       ov7670_pclk,
    input  logic       ov7670_href,
    input  logic       ov7670_v_sync,
    input  logic [7:0] ov7670_data,
    // piezo
    input  logic       piezo_stop,
    output logic       buzz,
    // export signals
    output logic       h_sync,
    output logic       v_sync,
    output logic [3:0] red_port,
    output logic [3:0] green_port,
    output logic [3:0] blue_port,
    output logic       SCL,
    output logic       SDA,
    output logic       slave_done,
    output logic [7:0] color_led,
    output logic [2:0] led_comp
);
    logic        we;
    logic [16:0] wAddr;
    logic [15:0] wData;
    logic [16:0] rAddr;
    logic [15:0] rData;
    logic [9:0] x_pixel, y_pixel;
    logic DE;
    logic w_rclk, rclk, oe, o_btn, o_piezo_stop;
    logic o_btn_game_start;

    logic [3:0] red_mem, green_mem, blue_mem;
    logic [3:0] red_chroma, green_chroma, blue_chroma;
    logic [3:0] red_back, green_back, blue_back;

    logic countdown_done;
    logic [7:0] detected_color;
    logic color_done;
    logic [1:0] crct_incrct;
    logic comparison_done;
    logic give_comparison_done;

    logic [1:0] startcount;
    logic [3:0] countdown;
    logic [1:0] win_lose_piezo;
    logic bgm_enable;



    VGA_Controller U_VGAController (
        .clk    (clk),
        .reset  (reset),
        .rclk   (w_rclk),
        .h_sync (h_sync),
        .v_sync (v_sync),
        .DE     (DE),
        .x_pixel(x_pixel),
        .y_pixel(y_pixel)
    );

    pixel_clk_gen U_OV7670_Clk_Gen (
        .clk  (clk),
        .reset(reset),
        .pclk (ov7670_xclk)
    );

    OV7670_MemController U_OV7670_MemController (
        .pclk       (ov7670_pclk),
        .reset      (reset),
        .href       (ov7670_href),
        .v_sync     (ov7670_v_sync),
        .ov7670_data(ov7670_data),
        .we         (we),
        .wAddr      (wAddr),
        .wData      (wData)
    );

    frame_buffer U_FrameBuffer (
        .wclk (ov7670_pclk),
        .we   (we),
        .wAddr(wAddr),
        .wData(wData),
        .rclk (rclk),
        .oe   (oe),
        .rAddr(rAddr),
        .rData(rData)
    );

    // QVGA_MemController U_QVGA_MemController (
    //     .clk       (w_rclk),
    //     .x_pixel   (x_pixel),
    //     .y_pixel   (y_pixel),
    //     .DE        (DE),
    //     .upscale   (upscale),
    //     .rclk      (rclk),
    //     .d_en      (oe),
    //     .rAddr     (rAddr),
    //     .rData     (rData),
    //     .red_port  (red_mem),    // 이것들 
    //     .green_port(green_mem),
    //     .blue_port (blue_mem)
    // );

    QVGA_MemController U_QVGA_MemController (
        .clk       (w_rclk),
        .pclk      (ov7670_xclk),  // 이거 왜 빠져있음?? // 이거 넣으니까 출력이 아예 안나오네
        .x_pixel   (x_pixel),
        .y_pixel   (y_pixel),
        .DE        (DE),
        .upscale   (upscale),
        .median    (median),
        .rclk      (rclk),
        .d_en      (oe),
        .rAddr     (rAddr),
        .rData     (rData),
        .red_port  (red_mem),      // 이것들 
        .green_port(green_mem),
        .blue_port (blue_mem)
    );

    chromakey_simple U_Chromakey (
        .clk         (w_rclk),
        .reset       (reset),
        .de          (DE),
        .mode_chroma (mode_chroma),
        .red_mem     (red_mem),
        .green_mem   (green_mem),
        .blue_mem    (blue_mem),
        .red_back    (red_back),
        .green_back  (green_back),
        .blue_back   (blue_back),
        .red_chroma  (red_chroma),
        .green_chroma(green_chroma),
        .blue_chroma (blue_chroma)
    );

    ImageRom U_ImageRom (
        .x_pixel   (x_pixel),
        .y_pixel   (y_pixel),
        .DE        (DE),
        .red_port  (red_back),
        .green_port(green_back),
        .blue_port (blue_back)
    );

    btn_debounce U_Btn_Debounce (
        .clk  (clk),
        .reset(reset),
        .i_btn(btn_sccb),
        .o_btn(o_btn)
    );

    SCCB_intf U_SCCB (
        .clk     (clk),
        .reset   (reset),
        .startSig(o_btn),
        .SCL     (SCL),
        .SDA     (SDA)
    );

    // display_lev1 U_Lev1 (
    //     .test_led(test_led),
    //     .pclk(ov7670_xclk),
    //     .reset(reset),

    //     // 이따가 바꾸기기

    //     .x_pixel(x_pixel),
    //     .y_pixel(y_pixel),
    //     .cam_r  (red_mem),
    //     .cam_g  (green_mem),
    //     .cam_b  (blue_mem),


    //     // export side
    //     .slave_done(slave_done),
    //     .red_port  (red_port),
    //     .green_port(green_port),
    //     .blue_port (blue_port),

    //     //Control Board side
    //     .sw(!q_start),
    //     .lvl(lvl),
    //     .question(question),
    //     .give_done(give_done),

    //     // Color Detect side
    //     .countdown_done(countdown_done),
    //     .startcount(startcount),
    //     .countdown(countdown),
    //     .bgm_enable(bgm_enable),



    //     // Color Comparison side
    //     .comparison_done(comparison_done),
    //     .give_comparison_done(give_comparison_done),
    //     .win_lose(crct_incrct)
    // );

    display_lev1 U_Lev1 (
        .test_led(test_led),
        .pclk    (ov7670_xclk),
        .reset   (reset),

        // 이따가 바꾸기기

        .x_pixel(x_pixel),
        .y_pixel(y_pixel),
        .cam_r  (red_chroma),
        .cam_g  (green_chroma),
        .cam_b  (blue_chroma),


        // export side
        .slave_done(slave_done),
        .red_port  (red_port),
        .green_port(green_port),
        .blue_port (blue_port),

        //Control Board side
        .sw       (!q_start),
        .lvl      (lvl),
        .question (question),
        .give_done(give_done),

        // Color Detect side
        .countdown_done(countdown_done),
        .startcount    (startcount),
        .countdown     (countdown),
        .bgm_enable    (bgm_enable),



        // Color Comparison side
        .comparison_done     (comparison_done),
        .give_comparison_done(give_comparison_done),
        .win_lose            (crct_incrct)
    );



    color_detect U_color_detect (
        .lvl            (lvl),
        // VGA Controller side
        .pclk           (ov7670_xclk),
        .reset          (reset),
        // btn
        .countdown_done (countdown_done),
        // x, y
        .x_pixel        (x_pixel),
        .y_pixel        (y_pixel),
        // export side    
        .red_port       (red_mem),
        .green_port     (green_mem),
        .blue_port      (blue_mem),
        .detected_color (detected_color),
        .comparison_done(comparison_done),
        .color_done     (color_done),
        .color_led      (color_led)

    );



    color_comparison U_color_comparison (
        // .lvl(lvl),  
        .pclk          (ov7670_xclk),
        .reset         (reset),
        .color_done    (color_done),
        .question      (question),        // 0:yellow 1:red
        .detected_color(detected_color),
        .led_comp      (led_comp),

        // GUI side
        .crct_incrct   (crct_incrct),
        .comparison_done(comparison_done),
        .give_comparison_done(give_comparison_done)
    );


    PIEZO U_PIEZO (
        .clk           (clk),
        .reset         (reset),
        .piezo_stop    (o_piezo_stop),
        .startcount    (startcount),    // display countdown 값
        .countdown     (countdown),     // display countdown 값
        .win_lose_piezo(crct_incrct),   // color_comparison 신호 받아옴
        .bgm_enable    (bgm_enable),    // IDLE bgm en
        .buzz          (buzz)
    );

    btn_debounce U_Btn_Debounce_piezo (
        .clk  (clk),
        .reset(reset),
        .i_btn(piezo_stop),
        .o_btn(o_piezo_stop)
    );


endmodule
