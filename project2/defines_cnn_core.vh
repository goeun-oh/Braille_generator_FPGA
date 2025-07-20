// -------------------------------------------------------------
// Global parameters –– 조정 시 여기만 변경하면 전체가 자동 변경
// -------------------------------------------------------------
`define IF_BW           8          // 입력 feature bit-width
`define W_BW            7          // 가중치 bit-width
`define BIAS_BW         6          // bias bit-width
`define OF_BW           32         // 풀링 이후 feature bit-width
`define ACC_BW          45         // FC 누적기 bit-width (=IF+W+log2(노드))

`define CI              3          // 입력 채널 수
`define CO              3          // 풀링 출력 채널 수 (MaxPool 뒤 채널은 변화 없음)
`define X_SIZE          12         // Conv2 출력 하나의 X, Y 크기
`define POOL_K          2          // MaxPool 2×2
`define P_SIZE          (`X_SIZE/`POOL_K) // 풀링 후 한 변 크기 (6×6)
`define FLAT_SIZE       (`CI*`P_SIZE*`P_SIZE) // 평탄화 총 길이 (=3×6×6=108)

`define FC_OUT          1          // 두 번째 이미지 예시에서는 1개의 출력 벡터