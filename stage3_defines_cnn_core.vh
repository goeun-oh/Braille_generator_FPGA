//======================================================================
// Global Parameters for CNN Design
// (Max Pooling -> Flatten -> Fully-Connected)
//======================================================================

//----------------------------------------------------------------------
// 1. 데이터 비트 폭 (Bit-Widths)
//----------------------------------------------------------------------
`define IF_BW           34  // 입력 Feature Map 픽셀 비트폭 (Max Pool 입력)
`define OF_BW           34  // 출력 Feature Map 픽셀 비트폭 (Max Pool 출력, FC 입력)
`define W_BW             8  // FC Layer 가중치(Weight) 비트폭
`define BIAS_BW          16  // FC Layer 편향(Bias) 비트폭

// 연산 과정에서의 비트 폭
`define MUL_BW          (`OF_BW + `W_BW)            // 곱셈 결과 비트폭 (34 + 8 = 42)
`define KER_BW          (`MUL_BW + $clog2(`pool_CO))
`define ACC_BW          (`MUL_BW + $clog2(`FC_IN_VEC)) // 내적 누산기 비트폭 (42 + log2(48) -> 42 + 6 = 48)
`define OUT_BW          (`ACC_BW + 1)               // 최종 출력 뉴런 비트폭 (누산기 + Bias 덧셈 후: 48 + 1 = 49)

//----------------------------------------------------------------------
// 2. 레이어 차원 (Layer Dimensions)
//----------------------------------------------------------------------
`define stage2_CI               3  // stage2 Relu output 채널 수
`define linebuf_CO              3  // linebuffer output 채널 수
`define pool_CI                 3  // pooling 입력 채널 수 linebuff 출력 채널과 같음
`define pool_CO                 3  // pooling 출력 채널 수 acc 입력 채널, kernal 입력 채널과 같음
`define acc_CO                  3  // acc 출력 채널 수 kernal 출력 채널, Core 입력 채널과 같음 
`define core_CO                 3  // core 출력 채널 수

//----------------------------------------------------------------------
// 3. Max Pooling 레이어 관련 파라미터
//----------------------------------------------------------------------
`define POOL_IN_SIZE     8  // Max Pool 입력 Feature Map의 한 변 크기 (8x8)
`define POOL_K           2  // Max Pool 커널 및 스트라이드 크기 (2x2)
`define P_SIZE          (`POOL_IN_SIZE / `POOL_K) // Max Pool 출력 Feature Map 한 변 크기 (4x4)

//----------------------------------------------------------------------
// 4. Fully-Connected (FC) 레이어 관련 파라미터
//----------------------------------------------------------------------
// FC 입력 벡터 길이 (Flatten 후)
`define FC_IN_VEC        (`pool_CO * `P_SIZE * `P_SIZE)   // 3 * 4 * 4 = 48

// Stride 비율
`define STRIDE           2
