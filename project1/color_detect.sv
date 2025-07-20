`timescale 1ns / 1ps
// 3color - red / yellow / blue
// red(100) / yellow(010) / blue(001) / etc(111)
// 현재 계획 datashift 방식으로 나누고 저장도 datashift 방식으로 수정하기기
// 디텍트 수정본

module color_detect (
    // level
    input  logic [1:0] lvl,
    // VGA Controller side
    input  logic       pclk,
    input  logic       reset,
    // countdown_done
    input  logic       countdown_done,
    input  logic       comparison_done,
    // x, y
    input  logic [9:0] x_pixel,
    input  logic [9:0] y_pixel,
    // export side    
    input  logic [3:0] red_port,
    input  logic [3:0] green_port,
    input  logic [3:0] blue_port,
    // 색깔 번호에 맞게 8bit 출력
    output logic [7:0] detected_color,
    // done signal
    output logic       color_done,
    // 색깔 확인
    output logic [7:0] color_led
);

    logic [10:0] red_sum0, red_next0;
    logic [10:0] green_sum0, green_next0;
    logic [10:0] blue_sum0, blue_next0;

    logic [10:0] red_sum1, red_next1;
    logic [10:0] green_sum1, green_next1;
    logic [10:0] blue_sum1, blue_next1;

    logic [10:0] red_sum2, red_next2;
    logic [10:0] green_sum2, green_next2;
    logic [10:0] blue_sum2, blue_next2;

    logic [10:0] red_sum3, red_next3;
    logic [10:0] green_sum3, green_next3;
    logic [10:0] blue_sum3, blue_next3;

    // 비트 수정
    logic [7:0] color_next;
    // logic [7:0] color_led_reg;
    logic [7:0] color_led_next;
    // 추가
    logic [3:0] avg_r0, avg_g0, avg_b0;
    logic [3:0] avg_r1, avg_g1, avg_b1;
    logic [3:0] avg_r2, avg_g2, avg_b2;
    logic [3:0] avg_r3, avg_g3, avg_b3;

    logic [3:0] maxC0, minC0, sat0;
    logic valid0, bright_red0, dark_red0, bright_yellow0, dark_yellow0, bright_blue0, dark_blue0;

    logic [3:0] maxC1, minC1, sat1;
    logic valid1, bright_red1, dark_red1, bright_yellow1, dark_yellow1, bright_blue1, dark_blue1;

    logic [3:0] maxC2, minC2, sat2;
    logic valid2, bright_red2, dark_red2, bright_yellow2, dark_yellow2, bright_blue2, dark_blue2;

    logic [3:0] maxC3, minC3, sat3;
    logic valid3, bright_red3, dark_red3, bright_yellow3, dark_yellow3, bright_blue3, dark_blue3;

    logic valid0y, valid1y, valid2y, valid3y;



    logic [6:0] state, next;

    logic display_en;

    logic tick_reg;

    // level 저장
    logic [1:0] level, level_next;

    parameter IDLE = 0, start_wait = 1, start0 = 2, save0 = 3, start1 = 4, save1_1 = 5, save1_0 = 6, start2 = 7, save2_2 = 8, save2_1 = 9, save2_0 = 10, start3 = 11, save3_3 = 12, save3_2 = 13, save3_1 = 14, save3_0 = 15, WAIT = 16;

    localparam BRIGHT_TH = 8;
    localparam DARK_TH = 8;
    localparam SAT_TH = 2;

    // localparam SAT_Y_TH = 3;
    localparam SAT_Y_TH = 1;

    // 밝기 기준도 낮춰보기 
    // // 변경 예시: R·G 둘 다 8 이상만 되어도 노랑으로 인식
    // localparam BRIGHT_TH = 8;


    always_ff @(posedge pclk, posedge reset) begin
        if (reset) begin
            state          <= 0;
            red_sum0       <= 0;
            green_sum0     <= 0;
            blue_sum0      <= 0;
            red_sum1       <= 0;
            green_sum1     <= 0;
            blue_sum1      <= 0;
            red_sum2       <= 0;
            green_sum2     <= 0;
            blue_sum2      <= 0;
            red_sum3       <= 0;
            green_sum3     <= 0;
            blue_sum3      <= 0;
            detected_color <= 0;
            color_done     <= 0;
            color_led      <= 0;
            level          <= 0;
        end else begin
            state          <= next;
            red_sum0       <= red_next0;
            green_sum0     <= green_next0;
            blue_sum0      <= blue_next0;
            red_sum1       <= red_next1;
            green_sum1     <= green_next1;
            blue_sum1      <= blue_next1;
            red_sum2       <= red_next2;
            green_sum2     <= green_next2;
            blue_sum2      <= blue_next2;
            red_sum3       <= red_next3;
            green_sum3     <= green_next3;
            blue_sum3      <= blue_next3;
            color_done     <= tick_reg;
            color_led      <= color_led_next;
            // 검출된 color
            detected_color <= color_next;
            level          <= level_next;
        end
    end

    /*
    lvl에 따른 코드 변화 짜기
    0 중앙에 하나
    1 그 줄에 2개
    2 그 줄에 3개
    3 그 줄에 4개

    // level
    input  logic [1:0] lvl,
    // level 변경 신호
    input  logic       q_start,
    */
    always_comb begin
        next          = state;
        color_next    = detected_color;
        red_next0     = red_sum0;
        green_next0   = green_sum0;
        blue_next0    = blue_sum0;
        red_next1     = red_sum1;
        green_next1   = green_sum1;
        blue_next1    = blue_sum1;
        red_next2     = red_sum2;
        green_next2   = green_sum2;
        blue_next2    = blue_sum2;
        red_next3     = red_sum3;
        green_next3   = green_sum3;
        blue_next3    = blue_sum3;
        tick_reg      = color_done;
        color_led_next = color_led;
        level_next    = level;
        case (state)
            IDLE: begin
                color_next    = 8'b0;
                // color_led_next = 8'b0;
                // color_led_next = 8'hFF;
                if (countdown_done) begin
                    next = start_wait;
                    level_next = lvl;
                end
            end
            /*
            display_en case 문 안에 lvl에 따라서 달라지도록 짜기
            y_pixel 고정 337 ~ 344
            x_pixel lvl0 316 ~ 324
            x_pixel lvl1 206 ~ 214  426 ~ 434
            x_pixel lvl2 126 ~ 134  316 ~ 324  506 ~ 514
            */
            start_wait: begin
                red_next0   = 0;
                green_next0 = 0;
                blue_next0  = 0;
                red_next1   = 0;
                green_next1 = 0;
                blue_next1  = 0;
                red_next2   = 0;
                green_next2 = 0;
                blue_next2  = 0;
                red_next3   = 0;
                green_next3 = 0;
                blue_next3  = 0;
                color_next  = 0;
                color_led_next = 0;
                // color_led_next = 8'hFF;
                if (y_pixel == 337 && x_pixel == 0) begin
                    case (level)
                        2'd0: begin
                            next = start0;
                        end
                        2'd1: begin
                            next = start1;
                        end
                        2'd2: begin
                            next = start2;
                        end
                        2'd3: begin
                            next = start3;
                        end
                    endcase
                end
            end
            start0: begin
                if ((x_pixel >= 317 && x_pixel <= 324) && (y_pixel >= 337 && y_pixel <= 344)) begin
                    red_next0   = red_sum0 + red_port;
                    green_next0 = green_sum0 + green_port;
                    blue_next0  = blue_sum0 + blue_port;
                end
                if (y_pixel > 344) next = save0;
            end
            save0: begin
                // 샘플링된 합산값으로 평균 계산
                avg_r0         = red_sum0 >> 6;
                avg_g0         = green_sum0 >> 6;
                avg_b0         = blue_sum0 >> 6;

                // 밝기(max), 채도(saturation) 계산
                // maxC0 : 평균 중 최댓값
                // minC0 : 평균 중 최솟값
                // sat0 : 색의 진하기(채도)
                maxC0          = (avg_r0 > avg_g0 && avg_r0 > avg_b0) ? avg_r0 : (avg_g0 > avg_b0) ? avg_g0 : avg_b0;
                minC0          = (avg_r0 < avg_g0 && avg_r0 < avg_b0) ? avg_r0 : (avg_g0 < avg_b0) ? avg_g0 : avg_b0;
                sat0           = maxC0 - minC0;

                // 유효 색상 검출 (채도 기준)
                valid0         = (sat0 > SAT_TH);  // 채도가 작으면 회색/노이즈로 보고 이후 검출을 건너뛴다.

                valid0y         = (sat0 > SAT_Y_TH);  // 채도가 작으면 회색/노이즈로 보고 이후 검출을 건너뛴다.
                
                // 밝은/어두운 색상 검출
                bright_red0    = valid0 && (avg_r0 > BRIGHT_TH) && (avg_r0 > avg_g0 + 1) && (avg_r0 > avg_b0 + 1);
                dark_red0      = valid0 && (maxC0 < DARK_TH) && (avg_r0 > avg_g0) && (avg_r0 > avg_b0);

                bright_yellow0 = valid0y && (avg_r0 > BRIGHT_TH) && (avg_g0 > BRIGHT_TH) && ((avg_r0 > avg_b0 + 1) && (avg_g0 > avg_b0 + 1)) && ((avg_r0 > avg_g0 ? avg_r0 - avg_g0 : avg_g0 - avg_r0) < 4);
                dark_yellow0   = valid0y && (maxC0 < DARK_TH) && (avg_r0 > avg_b0) && (avg_g0 > avg_b0) && ((avg_r0 > avg_g0 ? avg_r0 - avg_g0 : avg_g0 - avg_r0) < 4);

                bright_blue0   = valid0 && (avg_b0 > BRIGHT_TH) && (avg_b0 > avg_r0 + 1) && (avg_b0 > avg_g0 + 1);
                dark_blue0     = valid0 && (maxC0 < DARK_TH) && (avg_b0 > avg_r0) && (avg_b0 > avg_g0);

                // 최종 판정
                // 밝은/어두운 조건 중 하나라도 만족하면 기록, 어느 것도 아니면 00 처리
                if (bright_red0 || dark_red0) begin
                    color_next = {detected_color[5:0], 2'b01};
                    color_led_next = {color_led[5:0], 2'b01};
                end else if (bright_yellow0 || dark_yellow0) begin
                    color_next = {detected_color[5:0], 2'b10};
                    color_led_next = {color_led[5:0], 2'b10};
                end else if (bright_blue0 || dark_blue0) begin
                    color_next = {detected_color[5:0], 2'b11};
                    color_led_next = {color_led[5:0], 2'b11};
                end else begin
                    color_next = {detected_color[5:0], 2'b00};
                    color_led_next = {color_led[5:0], 2'b00};
                end

                tick_reg = 1;
                next     = WAIT;
                // color_led_next = 8'hFF;  // 내가 추가함


            end

            start1: begin
                if ((x_pixel >= 207 && x_pixel <= 214) && (y_pixel >= 337 && y_pixel <= 344)) begin
                    red_next0   = red_sum0 + red_port;
                    green_next0 = green_sum0 + green_port;
                    blue_next0  = blue_sum0 + blue_port;
                end else if ((x_pixel >= 427 && x_pixel <= 434) && (y_pixel >= 337 && y_pixel <= 344)) begin
                    red_next1   = red_sum1 + red_port;
                    green_next1 = green_sum1 + green_port;
                    blue_next1  = blue_sum1 + blue_port;
                end
                if (y_pixel > 344) next = save1_1;
            end
            save1_1: begin
                // 샘플링된 합산값으로 평균 계산
                avg_r1         = red_sum1 >> 6;
                avg_g1         = green_sum1 >> 6;
                avg_b1         = blue_sum1 >> 6;

                // 밝기(max), 채도(saturation) 계산
                // maxC0 : 평균 중 최댓값
                // minC0 : 평균 중 최솟값
                // sat0 : 색의 진하기(채도)
                maxC1          = (avg_r1 > avg_g1 && avg_r1 > avg_b1) ? avg_r1 : (avg_g1 > avg_b1) ? avg_g1 : avg_b1;
                minC1          = (avg_r1 < avg_g1 && avg_r1 < avg_b1) ? avg_r1 : (avg_g1 < avg_b1) ? avg_g1 : avg_b1;
                sat1           = maxC1 - minC1;

                // 유효 색상 검출 (채도 기준)
                valid1         = (sat1 > SAT_TH);  // 채도가 작으면 회색/노이즈로 보고 이후 검출을 건너뛴다.
                valid1y         = (sat1 > SAT_Y_TH);

                // 밝은/어두운 색상 검출
                bright_red1    = valid1 && (avg_r1 > BRIGHT_TH) && (avg_r1 > avg_g1 + 1) && (avg_r1 > avg_b1 + 1);
                dark_red1      = valid1 && (maxC1 < DARK_TH) && (avg_r1 > avg_g1) && (avg_r1 > avg_b1);

                bright_yellow1 = valid1y && (avg_r1 > BRIGHT_TH) && (avg_g1 > BRIGHT_TH) && ((avg_r1 > avg_b1 + 1) && (avg_g1 > avg_b1 + 1)) && ((avg_r1 > avg_g1 ? avg_r1 - avg_g1 : avg_g1 - avg_r1) < 4);
                dark_yellow1   = valid1y && (maxC1 < DARK_TH) && (avg_r1 > avg_b1) && (avg_g1 > avg_b1) && ((avg_r1 > avg_g1 ? avg_r1 - avg_g1 : avg_g1 - avg_r1) < 4);

                bright_blue1   = valid1 && (avg_b1 > BRIGHT_TH) && (avg_b1 > avg_r1 + 1) && (avg_b1 > avg_g1 + 1);
                dark_blue1     = valid1 && (maxC1 < DARK_TH) && (avg_b1 > avg_r1) && (avg_b1 > avg_g1);

                // 최종 판정
                // 밝은/어두운 조건 중 하나라도 만족하면 기록, 어느 것도 아니면 00 처리
                if (bright_red1 || dark_red1) begin
                    color_next = {detected_color[5:0], 2'b01};
                    color_led_next = {color_led[5:0], 2'b01};
                end else if (bright_yellow1 || dark_yellow1) begin
                    color_next = {detected_color[5:0], 2'b10};
                    color_led_next = {color_led[5:0], 2'b10};
                end else if (bright_blue1 || dark_blue1) begin
                    color_next = {detected_color[5:0], 2'b11};
                    color_led_next = {color_led[5:0], 2'b11};
                end else begin
                    color_next = {detected_color[5:0], 2'b00};
                    color_led_next = {color_led[5:0], 2'b00};
                end

                // tick_reg = 1;
                next = save1_0;

            end
            save1_0: begin
                // 샘플링된 합산값으로 평균 계산
                avg_r0         = red_sum0 >> 6;
                avg_g0         = green_sum0 >> 6;
                avg_b0         = blue_sum0 >> 6;

                // 밝기(max), 채도(saturation) 계산
                // maxC0 : 평균 중 최댓값
                // minC0 : 평균 중 최솟값
                // sat0 : 색의 진하기(채도)
                maxC0          = (avg_r0 > avg_g0 && avg_r0 > avg_b0) ? avg_r0 : (avg_g0 > avg_b0) ? avg_g0 : avg_b0;
                minC0          = (avg_r0 < avg_g0 && avg_r0 < avg_b0) ? avg_r0 : (avg_g0 < avg_b0) ? avg_g0 : avg_b0;
                sat0           = maxC0 - minC0;

                // 유효 색상 검출 (채도 기준)
                valid0         = (sat0 > SAT_TH);  // 채도가 작으면 회색/노이즈로 보고 이후 검출을 건너뛴다.
                valid0y         = (sat0 > SAT_Y_TH);  // 채도가 작으면 회색/노이즈로 보고 이후 검출을 건너뛴다.

                // 밝은/어두운 색상 검출
                bright_red0    = valid0 && (avg_r0 > BRIGHT_TH) && (avg_r0 > avg_g0 + 1) && (avg_r0 > avg_b0 + 1);
                dark_red0      = valid0 && (maxC0 < DARK_TH) && (avg_r0 > avg_g0) && (avg_r0 > avg_b0);

                bright_yellow0 = valid0y && (avg_r0 > BRIGHT_TH) && (avg_g0 > BRIGHT_TH) && ((avg_r0 > avg_b0 + 1) && (avg_g0 > avg_b0 + 1)) && ((avg_r0 > avg_g0 ? avg_r0 - avg_g0 : avg_g0 - avg_r0) < 4);
                dark_yellow0   = valid0y && (maxC0 < DARK_TH) && (avg_r0 > avg_b0) && (avg_g0 > avg_b0) && ((avg_r0 > avg_g0 ? avg_r0 - avg_g0 : avg_g0 - avg_r0) < 4);

                bright_blue0   = valid0 && (avg_b0 > BRIGHT_TH) && (avg_b0 > avg_r0 + 1) && (avg_b0 > avg_g0 + 1);
                dark_blue0     = valid0 && (maxC0 < DARK_TH) && (avg_b0 > avg_r0) && (avg_b0 > avg_g0);

                // 최종 판정
                // 밝은/어두운 조건 중 하나라도 만족하면 기록, 어느 것도 아니면 00 처리
                if (bright_red0 || dark_red0) begin
                    color_next = {detected_color[5:0], 2'b01};
                    color_led_next = {color_led[5:0], 2'b01};
                end else if (bright_yellow0 || dark_yellow0) begin
                    color_next = {detected_color[5:0], 2'b10};
                    color_led_next = {color_led[5:0], 2'b10};
                end else if (bright_blue0 || dark_blue0) begin
                    color_next = {detected_color[5:0], 2'b11};
                    color_led_next = {color_led[5:0], 2'b11};
                end else begin
                    color_next = {detected_color[5:0], 2'b00};
                    color_led_next = {color_led[5:0], 2'b00};
                end

                tick_reg = 1;
                next     = WAIT;
            end
            /*
            display_en case 문 안에 lvl에 따라서 달라지도록 짜기
            y_pixel 고정 337 ~ 344
            x_pixel lvl0 316 ~ 324
            x_pixel lvl1 206 ~ 214  426 ~ 434
            x_pixel lvl2 126 ~ 134  316 ~ 324  506 ~ 514
            x_pixel lvl3 88 ~ 96  240 ~ 248  392 ~ 400  544 ~ 552
            */
            start2: begin
                if ((x_pixel >= 127 && x_pixel <= 134) && (y_pixel >= 337 && y_pixel <= 344)) begin
                    red_next0   = red_sum0 + red_port;
                    green_next0 = green_sum0 + green_port;
                    blue_next0  = blue_sum0 + blue_port;
                end else if ((x_pixel >= 317 && x_pixel <= 324) && (y_pixel >= 337 && y_pixel <= 344)) begin
                    red_next1   = red_sum1 + red_port;
                    green_next1 = green_sum1 + green_port;
                    blue_next1  = blue_sum1 + blue_port;
                end else if ((x_pixel >= 507 && x_pixel <= 514) && (y_pixel >= 337 && y_pixel <= 344)) begin
                    red_next2   = red_sum2 + red_port;
                    green_next2 = green_sum2 + green_port;
                    blue_next2  = blue_sum2 + blue_port;
                end
                if (y_pixel > 344) next = save2_2;
            end
            save2_2: begin
                // 샘플링된 합산값으로 평균 계산
                avg_r2         = red_sum2 >> 6;
                avg_g2         = green_sum2 >> 6;
                avg_b2         = blue_sum2 >> 6;

                // 밝기(max), 채도(saturation) 계산
                maxC2          = (avg_r2 > avg_g2 && avg_r2 > avg_b2) ? avg_r2 : (avg_g2 > avg_b2) ? avg_g2 : avg_b2;
                minC2          = (avg_r2 < avg_g2 && avg_r2 < avg_b2) ? avg_r2 : (avg_g2 < avg_b2) ? avg_g2 : avg_b2;
                sat2           = maxC2 - minC2;

                // 유효 색상 검출 (채도 기준)
                valid2         = (sat2 > SAT_TH);  // 채도가 작으면 회색/노이즈로 건너뜀
                valid2y         = (sat2 > SAT_Y_TH);  // 채도가 작으면 회색/노이즈로 건너뜀

                // 밝은/어두운 색상 검출
                bright_red2    = valid2 && (avg_r2 > BRIGHT_TH) && (avg_r2 > avg_g2 + 1) && (avg_r2 > avg_b2 + 1);
                dark_red2      = valid2 && (maxC2 < DARK_TH) && (avg_r2 > avg_g2) && (avg_r2 > avg_b2);

                bright_yellow2 = valid2y && (avg_r2 > BRIGHT_TH) && (avg_g2 > BRIGHT_TH) && ((avg_r2 > avg_b2 + 1) && (avg_g2 > avg_b2 + 1)) && ((avg_r2 > avg_g2 ? avg_r2 - avg_g2 : avg_g2 - avg_r2) < 4);
                dark_yellow2   = valid2y && (maxC2 < DARK_TH) && (avg_r2 > avg_b2) && (avg_g2 > avg_b2) && ((avg_r2 > avg_g2 ? avg_r2 - avg_g2 : avg_g2 - avg_r2) < 4);

                bright_blue2   = valid2 && (avg_b2 > BRIGHT_TH) && (avg_b2 > avg_r2 + 1) && (avg_b2 > avg_g2 + 1);
                dark_blue2     = valid2 && (maxC2 < DARK_TH) && (avg_b2 > avg_r2) && (avg_b2 > avg_g2);

                // 최종 판정
                if (bright_red2 || dark_red2) begin
                    color_next    = {detected_color[5:0], 2'b01};
                    color_led_next = {color_led[5:0], 2'b01};
                end else if (bright_yellow2 || dark_yellow2) begin
                    color_next    = {detected_color[5:0], 2'b10};
                    color_led_next = {color_led[5:0], 2'b10};
                end else if (bright_blue2 || dark_blue2) begin
                    color_next    = {detected_color[5:0], 2'b11};
                    color_led_next = {color_led[5:0], 2'b11};
                end else begin
                    color_next    = {detected_color[5:0], 2'b00};
                    color_led_next = {color_led[5:0], 2'b00};
                end

                next = save2_1;
            end




            save2_1: begin

                // 1) 샘플링된 합산값으로 평균 계산
                avg_r1         = red_sum1 >> 6;
                avg_g1         = green_sum1 >> 6;
                avg_b1         = blue_sum1 >> 6;

                // 2) 밝기(max), 채도(saturation) 계산
                maxC1          = (avg_r1 > avg_g1 && avg_r1 > avg_b1) ? avg_r1 : (avg_g1 > avg_b1) ? avg_g1 : avg_b1;
                minC1          = (avg_r1 < avg_g1 && avg_r1 < avg_b1) ? avg_r1 : (avg_g1 < avg_b1) ? avg_g1 : avg_b1;
                sat1           = maxC1 - minC1;

                // 3) 유효 색상 검출 (채도 기준)
                valid1         = (sat1 > SAT_TH);
                valid1y         = (sat1 > SAT_Y_TH);

                // 4) 밝은/어두운 색상 검출
                bright_red1    = valid1 && (avg_r1 > BRIGHT_TH) && (avg_r1 > avg_g1 + 1) && (avg_r1 > avg_b1 + 1);
                dark_red1      = valid1 && (maxC1 < DARK_TH) && (avg_r1 > avg_g1) && (avg_r1 > avg_b1);

                bright_yellow1 = valid1y && (avg_r1 > BRIGHT_TH) && (avg_g1 > BRIGHT_TH) && ((avg_r1 > avg_b1 + 1) && (avg_g1 > avg_b1 + 1)) && ((avg_r1 > avg_g1 ? avg_r1 - avg_g1 : avg_g1 - avg_r1) < 4);
                dark_yellow1   = valid1y && (maxC1 < DARK_TH) && (avg_r1 > avg_b1) && (avg_g1 > avg_b1) && ((avg_r1 > avg_g1 ? avg_r1 - avg_g1 : avg_g1 - avg_r1) < 4);

                bright_blue1   = valid1 && (avg_b1 > BRIGHT_TH) && (avg_b1 > avg_r1 + 1) && (avg_b1 > avg_g1 + 1);
                dark_blue1     = valid1 && (maxC1 < DARK_TH) && (avg_b1 > avg_r1) && (avg_b1 > avg_g1);

                // 5) 최종 판정
                if (bright_red1 || dark_red1) begin
                    color_next    = {detected_color[5:0], 2'b01};
                    color_led_next = {color_led[5:0], 2'b01};
                end else if (bright_yellow1 || dark_yellow1) begin
                    color_next    = {detected_color[5:0], 2'b10};
                    color_led_next = {color_led[5:0], 2'b10};
                end else if (bright_blue1 || dark_blue1) begin
                    color_next    = {detected_color[5:0], 2'b11};
                    color_led_next = {color_led[5:0], 2'b11};
                end else begin
                    color_next    = {detected_color[5:0], 2'b00};
                    color_led_next = {color_led[5:0], 2'b00};
                end

                // 다음 단계로
                next = save2_0;
            end

            save2_0: begin
                // 1) 샘플링된 합산값으로 평균 계산
                avg_r0         = red_sum0 >> 6;
                avg_g0         = green_sum0 >> 6;
                avg_b0         = blue_sum0 >> 6;

                // 2) 밝기(max), 채도(saturation) 계산
                maxC0          = (avg_r0 > avg_g0 && avg_r0 > avg_b0) ? avg_r0 : (avg_g0 > avg_b0) ? avg_g0 : avg_b0;
                minC0          = (avg_r0 < avg_g0 && avg_r0 < avg_b0) ? avg_r0 : (avg_g0 < avg_b0) ? avg_g0 : avg_b0;
                sat0           = maxC0 - minC0;

                // 3) 유효 색상 검출 (채도 기준)
                valid0         = (sat0 > SAT_TH);
                valid0y         = (sat0 > SAT_Y_TH);

                // 4) 밝은/어두운 색상 검출
                bright_red0    = valid0 && (avg_r0 > BRIGHT_TH) && (avg_r0 > avg_g0 + 1) && (avg_r0 > avg_b0 + 1);
                dark_red0      = valid0 && (maxC0 < DARK_TH) && (avg_r0 > avg_g0) && (avg_r0 > avg_b0);

                bright_yellow0 = valid0y && (avg_r0 > BRIGHT_TH) && (avg_g0 > BRIGHT_TH) && ((avg_r0 > avg_b0 + 1) && (avg_g0 > avg_b0 + 1)) && ((avg_r0 > avg_g0 ? avg_r0 - avg_g0 : avg_g0 - avg_r0) < 4);
                dark_yellow0   = valid0y && (maxC0 < DARK_TH) && (avg_r0 > avg_b0) && (avg_g0 > avg_b0) && ((avg_r0 > avg_g0 ? avg_r0 - avg_g0 : avg_g0 - avg_r0) < 4);

                bright_blue0   = valid0 && (avg_b0 > BRIGHT_TH) && (avg_b0 > avg_r0 + 1) && (avg_b0 > avg_g0 + 1);
                dark_blue0     = valid0 && (maxC0 < DARK_TH) && (avg_b0 > avg_r0) && (avg_b0 > avg_g0);

                // 5) 최종 판정
                if (bright_red0 || dark_red0) begin
                    color_next    = {detected_color[5:0], 2'b01};
                    color_led_next = {color_led[5:0], 2'b01};
                end else if (bright_yellow0 || dark_yellow0) begin
                    color_next    = {detected_color[5:0], 2'b10};
                    color_led_next = {color_led[5:0], 2'b10};
                end else if (bright_blue0 || dark_blue0) begin
                    color_next    = {detected_color[5:0], 2'b11};
                    color_led_next = {color_led[5:0], 2'b11};
                end else begin
                    color_next    = {detected_color[5:0], 2'b00};
                    color_led_next = {color_led[5:0], 2'b00};
                end

                // 6) 완료 플래그 및 상태 전이
                tick_reg = 1;
                next     = WAIT;
            end

            start3: begin
                if ((x_pixel >= 89 && x_pixel <= 96) && (y_pixel >= 337 && y_pixel <= 344)) begin
                    red_next0   = red_sum0 + red_port;
                    green_next0 = green_sum0 + green_port;
                    blue_next0  = blue_sum0 + blue_port;
                end else if ((x_pixel >= 241 && x_pixel <= 248) && (y_pixel >= 337 && y_pixel <= 344)) begin
                    red_next1   = red_sum1 + red_port;
                    green_next1 = green_sum1 + green_port;
                    blue_next1  = blue_sum1 + blue_port;
                end else if ((x_pixel >= 393 && x_pixel <= 400) && (y_pixel >= 337 && y_pixel <= 344)) begin
                    red_next2   = red_sum2 + red_port;
                    green_next2 = green_sum2 + green_port;
                    blue_next2  = blue_sum2 + blue_port;
                end else if ((x_pixel >= 545 && x_pixel <= 552) && (y_pixel >= 337 && y_pixel <= 344)) begin
                    red_next3   = red_sum3 + red_port;
                    green_next3 = green_sum3 + green_port;
                    blue_next3  = blue_sum3 + blue_port;
                end

                if (y_pixel > 344) next = save3_3;
            end

            save3_3: begin
                // 1) 샘플링된 합산값으로 평균 계산
                avg_r3         = red_sum3 >> 6;
                avg_g3         = green_sum3 >> 6;
                avg_b3         = blue_sum3 >> 6;

                // 2) 밝기(max), 채도(saturation) 계산
                maxC3          = (avg_r3 > avg_g3 && avg_r3 > avg_b3) ? avg_r3 : (avg_g3 > avg_b3) ? avg_g3 : avg_b3;
                minC3          = (avg_r3 < avg_g3 && avg_r3 < avg_b3) ? avg_r3 : (avg_g3 < avg_b3) ? avg_g3 : avg_b3;
                sat3           = maxC3 - minC3;

                // 3) 유효 색상 검출 (채도 기준)
                valid3         = (sat3 > SAT_TH);
                valid3y         = (sat3 > SAT_Y_TH);

                // 4) 밝은/어두운 색상 검출
                bright_red3    = valid3 && (avg_r3 > BRIGHT_TH) && (avg_r3 > avg_g3 + 1) && (avg_r3 > avg_b3 + 1);
                dark_red3      = valid3 && (maxC3 < DARK_TH) && (avg_r3 > avg_g3) && (avg_r3 > avg_b3);

                bright_yellow3 = valid3y && (avg_r3 > BRIGHT_TH) && (avg_g3 > BRIGHT_TH) && ((avg_r3 > avg_b3 + 1) && (avg_g3 > avg_b3 + 1)) && ((avg_r3 > avg_g3 ? avg_r3 - avg_g3 : avg_g3 - avg_r3) < 4);
                dark_yellow3   = valid3y && (maxC3 < DARK_TH) && (avg_r3 > avg_b3) && (avg_g3 > avg_b3) && ((avg_r3 > avg_g3 ? avg_r3 - avg_g3 : avg_g3 - avg_r3) < 4);

                bright_blue3   = valid3 && (avg_b3 > BRIGHT_TH) && (avg_b3 > avg_r3 + 1) && (avg_b3 > avg_g3 + 1);
                dark_blue3     = valid3 && (maxC3 < DARK_TH) && (avg_b3 > avg_r3) && (avg_b3 > avg_g3);

                // 5) 최종 판정
                if (bright_red3 || dark_red3) begin
                    color_next    = {detected_color[5:0], 2'b01};
                    color_led_next = {color_led[5:0], 2'b01};
                end else if (bright_yellow3 || dark_yellow3) begin
                    color_next    = {detected_color[5:0], 2'b10};
                    color_led_next = {color_led[5:0], 2'b10};
                end else if (bright_blue3 || dark_blue3) begin
                    color_next    = {detected_color[5:0], 2'b11};
                    color_led_next = {color_led[5:0], 2'b11};
                end else begin
                    color_next    = {detected_color[5:0], 2'b00};
                    color_led_next = {color_led[5:0], 2'b00};
                end

                // 6) 다음 단계로 전이
                next = save3_2;
            end

            save3_2: begin
                // 1) 샘플링된 합산값으로 평균 계산
                avg_r2         = red_sum2 >> 6;
                avg_g2         = green_sum2 >> 6;
                avg_b2         = blue_sum2 >> 6;

                // 2) 밝기(max), 채도(saturation) 계산
                maxC2          = (avg_r2 > avg_g2 && avg_r2 > avg_b2) ? avg_r2 : (avg_g2 > avg_b2) ? avg_g2 : avg_b2;
                minC2          = (avg_r2 < avg_g2 && avg_r2 < avg_b2) ? avg_r2 : (avg_g2 < avg_b2) ? avg_g2 : avg_b2;
                sat2           = maxC2 - minC2;

                // 3) 유효 색상 검출 (채도 기준)
                valid2         = (sat2 > SAT_TH);
                valid2y         = (sat2 > SAT_Y_TH);

                // 4) 밝은/어두운 색상 검출
                bright_red2    = valid2 && (avg_r2 > BRIGHT_TH) && (avg_r2 > avg_g2 + 1) && (avg_r2 > avg_b2 + 1);
                dark_red2      = valid2 && (maxC2 < DARK_TH) && (avg_r2 > avg_g2) && (avg_r2 > avg_b2);

                bright_yellow2 = valid2y && (avg_r2 > BRIGHT_TH) && (avg_g2 > BRIGHT_TH) && ((avg_r2 > avg_b2 + 1) && (avg_g2 > avg_b2 + 1)) && ((avg_r2 > avg_g2 ? avg_r2 - avg_g2 : avg_g2 - avg_r2) < 4);
                dark_yellow2   = valid2y && (maxC2 < DARK_TH) && (avg_r2 > avg_b2) && (avg_g2 > avg_b2) && ((avg_r2 > avg_g2 ? avg_r2 - avg_g2 : avg_g2 - avg_r2) < 4);

                bright_blue2   = valid2 && (avg_b2 > BRIGHT_TH) && (avg_b2 > avg_r2 + 1) && (avg_b2 > avg_g2 + 1);
                dark_blue2     = valid2 && (maxC2 < DARK_TH) && (avg_b2 > avg_r2) && (avg_b2 > avg_g2);

                // 5) 최종 판정
                if (bright_red2 || dark_red2) begin
                    color_next    = {detected_color[5:0], 2'b01};
                    color_led_next = {color_led[5:0], 2'b01};
                end else if (bright_yellow2 || dark_yellow2) begin
                    color_next    = {detected_color[5:0], 2'b10};
                    color_led_next = {color_led[5:0], 2'b10};
                end else if (bright_blue2 || dark_blue2) begin
                    color_next    = {detected_color[5:0], 2'b11};
                    color_led_next = {color_led[5:0], 2'b11};
                end else begin
                    color_next    = {detected_color[5:0], 2'b00};
                    color_led_next = {color_led[5:0], 2'b00};
                end

                // 6) 다음 단계로 전이
                next = save3_1;
            end

            save3_1: begin
                // 1) 샘플링된 합산값으로 평균 계산
                avg_r1         = red_sum1 >> 6;
                avg_g1         = green_sum1 >> 6;
                avg_b1         = blue_sum1 >> 6;

                // 2) 밝기(max), 채도(saturation) 계산
                maxC1          = (avg_r1 > avg_g1 && avg_r1 > avg_b1) ? avg_r1 : (avg_g1 > avg_b1) ? avg_g1 : avg_b1;
                minC1          = (avg_r1 < avg_g1 && avg_r1 < avg_b1) ? avg_r1 : (avg_g1 < avg_b1) ? avg_g1 : avg_b1;
                sat1           = maxC1 - minC1;

                // 3) 유효 색상 검출 (채도 기준)
                valid1         = (sat1 > SAT_TH);
                valid1y         = (sat1 > SAT_Y_TH);

                // 4) 밝은/어두운 색상 검출
                bright_red1    = valid1 && (avg_r1 > BRIGHT_TH) && (avg_r1 > avg_g1 + 1) && (avg_r1 > avg_b1 + 1);
                dark_red1      = valid1 && (maxC1 < DARK_TH) && (avg_r1 > avg_g1) && (avg_r1 > avg_b1);

                bright_yellow1 = valid1y && (avg_r1 > BRIGHT_TH) && (avg_g1 > BRIGHT_TH) && ((avg_r1 > avg_b1 + 1) && (avg_g1 > avg_b1 + 1)) && ((avg_r1 > avg_g1 ? avg_r1 - avg_g1 : avg_g1 - avg_r1) < 4);
                dark_yellow1   = valid1y && (maxC1 < DARK_TH) && (avg_r1 > avg_b1) && (avg_g1 > avg_b1) && ((avg_r1 > avg_g1 ? avg_r1 - avg_g1 : avg_g1 - avg_r1) < 4);

                bright_blue1   = valid1 && (avg_b1 > BRIGHT_TH) && (avg_b1 > avg_r1 + 1) && (avg_b1 > avg_g1 + 1);
                dark_blue1     = valid1 && (maxC1 < DARK_TH) && (avg_b1 > avg_r1) && (avg_b1 > avg_g1);

                // 5) 최종 판정
                if (bright_red1 || dark_red1) begin
                    color_next    = {detected_color[5:0], 2'b01};
                    color_led_next = {color_led[5:0], 2'b01};
                end else if (bright_yellow1 || dark_yellow1) begin
                    color_next    = {detected_color[5:0], 2'b10};
                    color_led_next = {color_led[5:0], 2'b10};
                end else if (bright_blue1 || dark_blue1) begin
                    color_next    = {detected_color[5:0], 2'b11};
                    color_led_next = {color_led[5:0], 2'b11};
                end else begin
                    color_next    = {detected_color[5:0], 2'b00};
                    color_led_next = {color_led[5:0], 2'b00};
                end

                // 6) 다음 단계로 전이
                next = save3_0;
            end

            save3_0: begin
                // 1) 샘플링된 합산값으로 평균 계산
                avg_r0         = red_sum0 >> 6;
                avg_g0         = green_sum0 >> 6;
                avg_b0         = blue_sum0 >> 6;

                // 2) 밝기(max), 채도(saturation) 계산
                maxC0          = (avg_r0 > avg_g0 && avg_r0 > avg_b0) ? avg_r0 : (avg_g0 > avg_b0) ? avg_g0 : avg_b0;
                minC0          = (avg_r0 < avg_g0 && avg_r0 < avg_b0) ? avg_r0 : (avg_g0 < avg_b0) ? avg_g0 : avg_b0;
                sat0           = maxC0 - minC0;

                // 3) 유효 색상 검출 (채도 기준)
                valid0         = (sat0 > SAT_TH);
                valid0y         = (sat0 > SAT_Y_TH);

                // 4) 밝은/어두운 색상 검출
                bright_red0    = valid0 && (avg_r0 > BRIGHT_TH) && (avg_r0 > avg_g0 + 1) && (avg_r0 > avg_b0 + 1);
                dark_red0      = valid0 && (maxC0 < DARK_TH) && (avg_r0 > avg_g0) && (avg_r0 > avg_b0);

                bright_yellow0 = valid0y && (avg_r0 > BRIGHT_TH) && (avg_g0 > BRIGHT_TH) && ((avg_r0 > avg_b0 + 1) && (avg_g0 > avg_b0 + 1)) && ((avg_r0 > avg_g0 ? avg_r0 - avg_g0 : avg_g0 - avg_r0) < 4);
                dark_yellow0   = valid0y && (maxC0 < DARK_TH) && (avg_r0 > avg_b0) && (avg_g0 > avg_b0) && ((avg_r0 > avg_g0 ? avg_r0 - avg_g0 : avg_g0 - avg_r0) < 4);

                bright_blue0   = valid0 && (avg_b0 > BRIGHT_TH) && (avg_b0 > avg_r0 + 1) && (avg_b0 > avg_g0 + 1);
                dark_blue0     = valid0 && (maxC0 < DARK_TH) && (avg_b0 > avg_r0) && (avg_b0 > avg_g0);

                // 5) 최종 판정
                if (bright_red0 || dark_red0) begin
                    color_next    = {detected_color[5:0], 2'b01};
                    color_led_next = {color_led[5:0], 2'b01};
                end else if (bright_yellow0 || dark_yellow0) begin
                    color_next    = {detected_color[5:0], 2'b10};
                    color_led_next = {color_led[5:0], 2'b10};
                end else if (bright_blue0 || dark_blue0) begin
                    color_next    = {detected_color[5:0], 2'b11};
                    color_led_next = {color_led[5:0], 2'b11};
                end else begin
                    color_next    = {detected_color[5:0], 2'b00};
                    color_led_next = {color_led[5:0], 2'b00};
                end

                // 6) 완료 플래그 및 상태 전이
                tick_reg = 1;
                next     = WAIT;
            end

            WAIT: begin
                tick_reg = 0;
                if (comparison_done) begin
                    next = IDLE;
                end
            end
        endcase

    end
endmodule
