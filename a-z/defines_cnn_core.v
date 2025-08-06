`define    KX			   5  // Number of Kernel X
`define    KY			   5  // Number of Kernel Y
//======================================================================
//          ISP
//======================================================================
`define ISP_BW            32

//======================================================================
//          stage 1
//======================================================================

`define ST1_IX            28
`define ST1_IY            28

`define ST1_I_F_BW        8 // Bit Width of Input Stage1
`define ST1_W_BW          8 // BW of weight parameter
`define ST1_B_BW          16 // BW of bias parameter
`define ST1_O_F_BW        `ST1_AR_BW
`define ST1_CI            1 
`define ST1_CO            3

`define ST1_I_M_BW        (`ST1_I_F_BW+`ST1_W_BW)       
`define ST1_AK_BW         (`ST1_I_M_BW + $clog2(`KX*`KY))         // ST1_M_BW + log(KY*KX) Accum Kernel 
`define ST1_ACI_BW        (`ST1_AK_BW + $clog2(`ST1_CI))          // Accum Channel Input
`define ST1_AB_BW         (`ST1_ACI_BW+1)                         // After add bias, 연산애매해서 일단 +1함
`define ST1_AR_BW         (`ST1_AB_BW-1)                          // After ReLU

`define ST1_OUT_W         (`ST1_IX - `KX + 1)
`define ST1_OUT_H         (`ST1_IY - `KY + 1)
`define ST1_POOL_OUT_W    12
`define ST1_POOL_OUT_H    12
`define TOTAL_PIXELS      (`ST1_IX * `ST1_IY)

`define ST1_BITSHIFT_BW   12
//======================================================================
//          stage 2
//======================================================================

`define    ST2_W_BW        8  // BW of weight parameter
`define    ST2_B_BW        16  // BW of bias parameter
`define    ST2_M_BW        (`ST2_Conv_IBW + `ST2_W_BW) // I_F_BW * W_BW
`define    ST2_AK_BW       (`ST2_M_BW + $clog2(`KY*`KX))    // ST2_M_BW + log(KY*KX) Accum Kernel 

`define    ST2_ACI_BW	   (`ST2_AK_BW + $clog2(`ST2_Conv_CI)) // ST2_AK_BW + log (CI) Accum Channel Input      35bit
`define    ST2_AB_BW       (`ST2_ACI_BW+1) // ST2_ACI_BW + bias (#1). // After add bias, 연산애매해서 일단 +1함   36bit
`define    ST2_O_F_BW      (`ST2_AB_BW-1) // reLU Activation,                                                   35bit


// `define    O_F_ACC_BW   27 // for demo, ST2_O_F_BW + log (CO)

//pooling interface
`define    ST2_Pool_IBW    (`ST1_O_F_BW-`ST1_BITSHIFT_BW) // Stage2 Pooling input bitwidth
`define    ST2_Pool_CI     3  // Number of Stage2 Pooling Channel Input
`define    ST2_Pool_CO     3  // Number of Stage2 Pooling Channel Output
`define    ST2_Pool_X      24 // Number of X (Input Channel)
`define    ST2_Pool_Y      24 // Number of y (Input Channel)

//convolution interface
`define    ST2_Conv_IBW    `ST2_Pool_IBW // Conv Input Bitwidth
`define    ST2_Conv_CI     3  // Number of Stage2 Conv Channel Input
`define    ST2_Conv_CO     6  // Number of Stage2 Conv Channel Output
`define    ST2_Conv_X      12 // Number of X (Conv Input Channel)
`define    ST2_Conv_Y      12 // Number of y (Con Input Channel)


`define    ST2_BITSHIFT_BW 8
//======================================================================
//          stage 3
//======================================================================

//======================================================================
// Global Parameters for CNN Design
// (Max Pooling -> Flatten -> Fully-Connected)
//======================================================================

//----------------------------------------------------------------------
// 1. 데이터 비트 폭 (Bit-Widths)
//----------------------------------------------------------------------

`define ST3_IF_BW           (`ST2_O_F_BW-`ST2_BITSHIFT_BW)  // 입력 Feature Map 픽셀 비트폭 (Max Pool 입력)
`define ST3_OF_BW           `ST3_IF_BW  // 출력 Feature Map 픽셀 비트폭 (Max Pool 출력, FC 입력)
`define ST3_W_BW             8  // FC Layer 가중치(Weight) 비트폭
`define ST3_BIAS_BW          16  // FC Layer 편향(Bias) 비트폭

// 연산 과정에서의 비트 폭
`define ST3_MUL_BW          (`ST3_OF_BW + `ST3_W_BW)            // 곱셈 결과 비트폭 (35 + 8 = 43)
`define ST3_KER_BW          (`ST3_MUL_BW + $clog2(`pool_CO))    // 45비트
`define ST3_ACC_BW          (`ST3_MUL_BW + $clog2(`FC_IN_VEC))  // 내적 누산기 비트폭 (45 + log2(48) -> 45 + 5.xx = 51)
`define ST3_OUT_BW          (`ST3_ACC_BW + 1 -(`ST3_BITSHIFT_BW)) // 최종 출력 뉴런 비트폭 (누산기 + Bias 덧셈 후: 51 + 1 = 52)


`define    ST3_BITSHIFT_BW  16
//----------------------------------------------------------------------
// 2. 레이어 차원 (Layer Dimensions)
//----------------------------------------------------------------------
`define stage3_CI               6  // stage2 Relu output 채널 수 (= stage3 CI)
`define linebuf_CO              `stage3_CI  // linebuffer output 채널 수
`define pool_CI                 `stage3_CI  // pooling 입력 채널 수 linebuff 출력 채널과 같음
`define pool_CO                 `stage3_CI  // pooling 출력 채널 수 acc 입력 채널, kernal 입력 채널과 같음
`define acc_CO                  26  // acc 출력 채널 수 kernal 출력 채널, Core 입력 채널과 같음 
`define core_CO                 26  // core 출력 채널 수

//----------------------------------------------------------------------
// 3. Max Pooling 레이어 관련 파라미터
//----------------------------------------------------------------------
`define POOL_IN_SIZE     (`ST2_Conv_X-`KX+1)  // Max Pool 입력 Feature Map의 한 변 크기 (8x8)
`define POOL_K           2  // Max Pool 커널 및 스트라이드 크기 (2x2)
`define P_SIZE           (`POOL_IN_SIZE / `POOL_K) // Max Pool 출력 Feature Map 한 변 크기 (4x4)

//----------------------------------------------------------------------
// 4. Fully-Connected (FC) 레이어 관련 파라미터
//----------------------------------------------------------------------
// FC 입력 벡터 길이 (Flatten 후)
`define FC_IN_VEC        (`pool_CO * `P_SIZE * `P_SIZE)   // 3 * 4 * 4 = 48

// Stride 비율
`define STRIDE           2
