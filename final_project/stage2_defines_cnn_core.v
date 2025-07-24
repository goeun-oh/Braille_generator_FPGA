
`define    KX			5  // Number of Kernel X
`define    KY			5  // Number of Kernel Y

`define    W_BW         8  // BW of weight parameter
`define    B_BW         8  // BW of bias parameter
`define    M_BW         28 // I_F_BW * W_BW
`define    AK_BW        33 // M_BW + log(KY*KX) Accum Kernel 

`define    ACI_BW		35 // AK_BW + log (CI) Accum Channel Input
`define    AB_BW        35 // ACI_BW + bias (#1). 
`define    O_F_BW       35 // reLU Activation, after Activation it becomes O_F_BW-1 bit

// `define    O_F_ACC_BW   27 // for demo, O_F_BW + log (CO)

//pooling interface
`define    ST2_Pool_IBW    20 // Stage2 Pooling input bitwidth
`define    ST2_Pool_CI     3  // Number of Stage2 Pooling Channel Input
`define    ST2_Pool_CO     3  // Number of Stage2 Pooling Channel Output
`define    ST2_Pool_X      24 // Number of X (Input Channel)
`define    ST2_Pool_Y      24 // Number of y (Input Channel)

//convolution interface
`define    ST2_Conv_IBW    20 // Conv Input Bitwidth
`define    ST2_Conv_CI     3  // Number of Stage2 Conv Channel Input
`define    ST2_Conv_CO     3  // Number of Stage2 Conv Channel Output
`define    ST2_Conv_X      12 // Number of X (Conv Input Channel)
`define    ST2_Conv_Y      12 // Number of y (Con Input Channel)
