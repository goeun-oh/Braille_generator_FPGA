`timescale 1ns / 1ps
/*
    PIEZO 모듈: 카트라이더 주제곡 메인 멜로디 반복 재생 + start/countdown beep + win/lose 멜로디
    - 배경 음악(bgm) 기능
    - win/lose 시 멜로디 재생 기능
    - start/countdown에서 BEEP 기능
    - piezo_stop 입력으로 전체 재생 토글
*/

// 음계 선택용 열거형 정의 (16가지 음 + 무음)
typedef enum logic [3:0] {
    C4   = 4'd0,   // 도 (C4)
    D4   = 4'd1,   // 레 (D4)
    E4   = 4'd2,   // 미 (E4)
    F4   = 4'd3,   // 파 (F4)
    G4   = 4'd4,   // 솔 (G4)
    A4   = 4'd5,   // 라 (A4)
    B4   = 4'd6,   // 시 (B4)
    C5   = 4'd7,   // 도 (C5, 한 옥타브 위 도)
    D5   = 4'd8,   // 레 (D5)
    E5   = 4'd9,   // 미 (E5)
    F5   = 4'd10,  // 파 (F5)
    G5   = 4'd11,  // 솔 (G5)
    A5   = 4'd12,  // 라 (A5)
    B5   = 4'd13,  // 시 (B5)
    C6   = 4'd14,  // 도 (C6, 두 옥타브 위 도)
    NONE = 4'd15   // 무음
} NoteSel;

module PIEZO (
    input  logic       clk,
    input  logic       reset,
    input  logic       piezo_stop,
    input  logic [1:0] startcount,      // 시작 카운트
    input  logic [3:0] countdown,       // 레벨 카운트다운
    input  logic [1:0] win_lose_piezo,  // 승리/패배
    input  logic       bgm_enable,      // 배경음악 재생 허용(IDLE)
    output logic       buzz             // 부저 출력
);


    // start/countdown에서 단일 beep 요청 처리
    logic [ 1:0] startcount_r;
    logic        start_edge;
    logic [ 3:0] countdown_r;
    logic        countdown_edge;
    logic        beep_req;
    logic [31:0] beep_cnt;

    // 엣지 검출: 값이 변경될 때만 트리거
    assign start_edge     = (startcount != startcount_r);
    assign countdown_edge = (countdown != countdown_r);

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            startcount_r <= 2'd0;
            countdown_r  <= 4'd0;
        end else begin
            startcount_r <= startcount;
            countdown_r  <= countdown;
        end
    end

    localparam integer BEEP_SHORT = 5_000_000;  // 짧은 BEEP 지속 시간 (0.05s)
    localparam integer BEEP_LONG = 20_000_000;  // 긴 BEEP 지속 시간 (0.20s)


    // start/countdown에서 단일 beep 요청 처리
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            beep_req <= 1'b0;
            beep_cnt <= 0;
        end  // startcount 엣지: 무조건 short/long 구분
        else if (start_edge) begin
            beep_req <= 1'b1;
            beep_cnt <= (startcount == 2'd0) ? BEEP_LONG : BEEP_SHORT;
        end  // countdown 엣지: 이전 카운트가 0이 아닐 때만 트리거
        else if (countdown_edge && countdown_r != 4'd0) begin
            beep_req <= 1'b1;
            // 도달한 값이 0이면 long, 그 외엔 short
            beep_cnt <= (countdown == 4'd0) ? BEEP_LONG : BEEP_SHORT;
        end  // 이미 요청된 beep 처리
        else if (beep_req) begin
            if (beep_cnt == 0) begin
                beep_req <= 1'b0;
            end else begin
                beep_cnt <= beep_cnt - 1;
            end
        end
    end



    // BEEP 톤 출력 (기본 TONE_HALF)
    localparam integer TONE_HALF = 20_000;
    logic [15:0] tone_cnt;
    logic        tone_out;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            tone_cnt <= 0;
            tone_out <= 1'b0;
        end else if (beep_req) begin
            if (tone_cnt >= TONE_HALF) begin
                tone_cnt <= 0;
                tone_out <= ~tone_out;
            end else tone_cnt <= tone_cnt + 1;
        end else begin
            tone_out <= 1'b0;  // BEEP가 아닐 땐 무음
        end
    end


    // 재생 토글 (piezo_stop)
    logic play_enable, piezo_stop_q;
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            play_enable  <= 1'b1;  // 초기 자동 재생
            piezo_stop_q <= 1'b0;
        end else begin
            piezo_stop_q <= piezo_stop;
            if (piezo_stop & ~piezo_stop_q) play_enable <= ~play_enable;  // 엣지마다 토글
        end
    end


    // win/lose 멜로디 정의 및 재생 FSM
    localparam integer WIN_LEN = 4;
    localparam integer LOSE_LEN = 4;
    NoteSel win_melody [ WIN_LEN] = '{C4, E4, G4, C5};  // 승리음: C4-E4-G4-C5
    NoteSel lose_melody[LOSE_LEN] = '{C5, B4, A4, G4};  // 패배음: C5-B4-A4-G4


    // 음표 길이 정의 (박자 단위)
    // localparam integer QTR     = 75_000_000;  // 1/4박자 = 0.75 s
    localparam integer QTR = 37_500_000;  // 1/4박자 = 0.375 s
    localparam integer EIG = QTR / 2;  // 1/8박자
    localparam integer SIX = QTR / 4;  // 1/16박자
    localparam integer HLF = QTR * 2;  // 1/2박자
    localparam integer DOT_QTR = QTR + EIG;  // 점4분음표 (1/4 + 1/8)
    localparam integer WHOLE = QTR * 4;  // 2박자(온음표)

    // 승/패 시작 & 인덱스/카운트 초기화, 진행 제어
    logic   [ 1:0] win_lose_piezo_r;
    logic   [ 1:0] melody_mode;
    logic   [ 1:0] melody_idx;
    logic   [31:0] melody_cnt;
    logic          melody_playing;
    NoteSel        melody_note;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            win_lose_piezo_r <= 2'b00;
            melody_mode      <= 2'b00;
            melody_idx       <= 2'd0;
            melody_cnt       <= 32'd0;
            melody_playing   <= 1'b0;
        end else begin
            win_lose_piezo_r <= win_lose_piezo;
            // win 시작
            if (win_lose_piezo == 2'b01 && win_lose_piezo_r != 2'b01) begin
                melody_mode    <= 2'b01;
                melody_playing <= 1'b1;
                melody_idx     <= 2'd0;
                melody_cnt     <= 32'd0;
            end  // lose 시작
            else if (win_lose_piezo == 2'b10 && win_lose_piezo_r != 2'b10) begin
                melody_mode    <= 2'b10;
                melody_playing <= 1'b1;
                melody_idx     <= 2'd0;
                melody_cnt     <= 32'd0;
            end  // 재생 중 음 다음으로 이동
            else if (melody_playing) begin
                if (melody_cnt >= QTR) begin
                    melody_cnt <= 32'd0;
                    melody_idx <= melody_idx + 1;
                    // 마지막 음 재생 후 종료
                    if ((melody_mode == 2'b01 && melody_idx + 1 >= WIN_LEN) || (melody_mode == 2'b10 && melody_idx + 1 >= LOSE_LEN)) begin
                        melody_playing <= 1'b0;
                        melody_mode    <= 2'b00;
                    end
                end else begin
                    melody_cnt <= melody_cnt + 1;
                end
            end
        end
    end

    // 현재 재생할 음 선택
    always_comb begin
        if (melody_playing && melody_mode == 2'b01) begin
            melody_note = win_melody[melody_idx];
        end else if (melody_playing && melody_mode == 2'b10) begin
            melody_note = lose_melody[melody_idx];
        end else begin
            melody_note = C4;
        end
    end


    // 메인 배경 멜로디 (KartRider 테마)
    localparam integer MAIN_LEN = 23;

    NoteSel main_melody [MAIN_LEN] = '{
        G5,  E5,  G5,  E5,
        A5,  G5,  F5,  E5,
        F5,  G5,  E5,  F5,
        G5,  E5,  G5,  E5,
        F5, E5, F5, E5,
        D5, B4, C5
    };


    integer main_dur [MAIN_LEN] = '{
           DOT_QTR,    EIG,    DOT_QTR,  EIG,
           EIG,    EIG,    EIG,  EIG,
           EIG,    QTR,    SIX,  SIX,
           DOT_QTR,    EIG,    DOT_QTR,  EIG,
           SIX, SIX, SIX, SIX, EIG, EIG, WHOLE
    };

    logic [4:0]  main_idx;
    logic [31:0] main_cnt;
    // wire         main_play = play_enable & ~melody_playing;
    wire main_play = play_enable & ~melody_playing & bgm_enable;

    // 메인 멜로디 진행 제어
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            main_idx <= 0;
            main_cnt <= 0;
        end else if (main_play) begin
            if (main_cnt >= main_dur[main_idx]) begin
                main_cnt <= 0;
                main_idx <= (main_idx == MAIN_LEN - 1) ? 0 : main_idx + 1;
            end else begin
                main_cnt <= main_cnt + 1;
            end
        end
    end

    // 최종 출력할 음(note) 결정
    // 우선순위: beep_req > win/lose 멜로디 > 메인멜로디 > 무음
    NoteSel current_note;

    always_comb begin
        if (beep_req) begin
            current_note = NONE;
        end else if (melody_playing) begin
            current_note = melody_note;
        end else if (main_play) begin
            current_note = main_melody[main_idx];
        end else begin
            current_note = NONE;  // 무음
        end
    end

    // current_note에 따른 주기(period) 결정
    localparam integer P_C4 = 190_839;  // 262Hz
    localparam integer P_D4 = 170_068;  // 294Hz
    localparam integer P_E4 = 151_515;  // 330Hz
    localparam integer P_F4 = 143_000;  // 349Hz
    localparam integer P_G4 = 127_551;  // 392Hz
    localparam integer P_A4 = 113_636;  // 440Hz
    localparam integer P_B4 = 101_217;  // 494Hz
    localparam integer P_C5 = 95_430;  // 523Hz
    localparam integer P_D5 = 85_034;  // 587Hz
    localparam integer P_E5 = 75_758;  // 659Hz
    localparam integer P_F5 = 71_500;  // 698Hz
    localparam integer P_G5 = 63_776;  // 784Hz
    localparam integer P_A5 = 56_818;  // 880Hz
    localparam integer P_B5 = 50_609;  // 988Hz
    localparam integer P_C6 = 47_715;  // 1047Hz

    logic [31:0] period;

    always_comb begin
        unique case (current_note)
            C4: period = P_C4;
            D4: period = P_D4;
            E4: period = P_E4;
            F4: period = P_F4;
            G4: period = P_G4;
            A4: period = P_A4;
            B4: period = P_B4;
            C5: period = P_C5;
            D5: period = P_D5;
            E5: period = P_E5;
            F5: period = P_F5;
            G5: period = P_G5;
            A5: period = P_A5;
            B5: period = P_B5;
            C6: period = P_C6;
            default: period = P_C4;
        endcase
    end

    // 최종 부저 신호 생성
    logic [31:0] counter;
    logic        buzz_int;

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            counter  <= 0;
            buzz_int <= 1'b0;
        end else if (play_enable && !beep_req && current_note != NONE) begin
            if (counter >= period) begin
                counter  <= 0;
                buzz_int <= ~buzz_int;
            end else begin
                counter <= counter + 1;
            end
        end else begin
            counter  <= 0;
            buzz_int <= 1'b0;
        end
    end

    // beep_req 우선, 그 외에는 buzz_int 출력
    assign buzz = beep_req ? tone_out : buzz_int;

endmodule
