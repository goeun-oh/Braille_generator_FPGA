// --- Global Parameters for This Specific Design Flow ---

// 1. Bit-Widths
`define IF_BW      32  // Input Feature Map Bit-Width (Max Pool 입력)
`define W_BW        7  // FC Layer Weight Bit-Width
`define BIAS_BW     6  // FC Layer Bias Bit-Width
`define OF_BW      32  // Output Feature Map Bit-Width (Max Pool 출력)
`define MUL_BW     (`OF_BW + `W_BW)           // 곱셈 결과 Bit-Width (32 + 7 = 39)
`define ACC_BW     45  // 누산기 Bit-Width (곱셈 결과 + log2(벡터길이))
`define OUT_BW     (`MUL_BW + 1)             // 최종 출력 Bit-Width (Bias 덧셈 후, 39 + 1 = 40)

// 2. Channel and Layer Dimensions
`define CI          3  // 입력 채널 수 (Input Channels)
`define CO          3  // 출력 채널 수 (Output Channels / FC Neurons)

// 3. Max Pooling Layer Dimensions
`define POOL_IN_SIZE  8  // Max Pool 입력 Feature Map의 한 변 크기 (8x8)
`define POOL_K        2  // Max Pool 커널 크기 (2x2)
`define P_SIZE      (`POOL_IN_SIZE / `POOL_K) // Max Pool 출력 Feature Map 한 변 크기 (4x4)

// 4. Fully-Connected (FC) Layer Dimensions
`define FC_IN_VEC   (`CI * `P_SIZE * `P_SIZE) // FC 입력 벡터 길이 (3 * 4 * 4 = 48)



// // -------------------------------------------------------------
// // Global parameters –– 조정 시 여기만 변경하면 전체가 자동 변경
// // -------------------------------------------------------------
// `define IF_BW           8          // 입력 feature bit-width
// `define W_BW            7          // 가중치 bit-width
// `define BIAS_BW         6          // bias bit-width
// `define OF_BW           32         // 풀링 이후 feature bit-width
// `define ACC_BW          45         // FC 누적기 bit-width (=IF+W+log2(노드))

// `define CI              3          // 입력 채널 수
// `define CO              3          // 풀링 출력 채널 수 (MaxPool 뒤 채널은 변화 없음)
// `define X_SIZE          12         // Conv2 출력 하나의 X, Y 크기
// `define POOL_K          2          // MaxPool 2×2
// `define P_SIZE          (`X_SIZE/`POOL_K) // 풀링 후 한 변 크기 (6×6)
// `define FLAT_SIZE       (`CI*`P_SIZE*`P_SIZE) // 평탄화 총 길이 (=3×6×6=108)

// `define FC_OUT          1          // 두 번째 이미지 예시에서는 1개의 출력 벡터