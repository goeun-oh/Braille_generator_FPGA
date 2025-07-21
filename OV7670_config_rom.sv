`timescale 1ns / 1ps
module OV7670_config_rom (
    input  logic        clk,
    input  logic [ 7:0] rom_addr,
    output logic [15:0] rom_data
);

    //FFFF is end of rom, FFF0 is delay
    always @(posedge clk) begin
        case (rom_addr)
            0: rom_data <= 16'h12_80;  //reset
            1: rom_data <= 16'hFF_F0;  //delay
            2:
            rom_data <= 16'h12_14;  // COM7,     set RGB color output and set QVGA
            3:
            rom_data <= 16'h11_80;  // CLKRC     internal PLL matches input clock
            4: rom_data <= 16'h0C_04;  // COM3,     default settings
            5: rom_data <= 16'h3E_19;  // COM14,    no scaling, normal pclock
            6: rom_data <= 16'h04_00;  // COM1,     disable CCIR656
            7: rom_data <= 16'h40_d0;  //COM15,     RGB565, full output range
            8: rom_data <= 16'h3a_04;  //TSLB       
            9: rom_data <= 16'h14_18;  //COM9       MAX AGC value x4
            10: rom_data <= 16'h4F_B3;  //MTX1       
            11: rom_data <= 16'h50_B3;  //MTX2
            12: rom_data <= 16'h51_00;  //MTX3
            13: rom_data <= 16'h52_3d;  //MTX4
            14: rom_data <= 16'h53_A7;  //MTX5
            15: rom_data <= 16'h54_E4;  //MTX6
            16: rom_data <= 16'h58_9E;  //MTXS
            17:
            rom_data <= 16'h3D_C0; //COM13      sets gamma enable, does not preserve reserved bits, may be wrong?
            18: rom_data <= 16'h17_15;  //HSTART     start high 8 bits 
            19:
            rom_data <= 16'h18_03; //HSTOP      stop high 8 bits //these kill the odd colored line
            20: rom_data <= 16'h32_00;  //91  //HREF       edge offset
            21: rom_data <= 16'h19_03;  //VSTART     start high 8 bits
            22: rom_data <= 16'h1A_7B;  //VSTOP      stop high 8 bits
            23: rom_data <= 16'h03_00;  // 00 //VREF       vsync edge offset
            24: rom_data <= 16'h0F_41;  //COM6       reset timings
            25:
            rom_data <= 16'h1E_00; //MVFP       disable mirror / flip //might have magic value of 03
            26:
            rom_data <= 16'h33_0B;  //CHLF       //magic value from the internet
            27: rom_data <= 16'h3C_78;  //COM12      no HREF when VSYNC low
            28: rom_data <= 16'h69_00;  //GFIX       fix gain control
            29: rom_data <= 16'h74_00;  //REG74      Digital gain control
            30:
            rom_data <= 16'hB0_84; //RSVD       magic value from the internet *required* for good color
            31: rom_data <= 16'hB1_0c;  //ABLC1
            32: rom_data <= 16'hB2_0e;  //RSVD       more magic internet values
            33: rom_data <= 16'hB3_80;  //THL_ST
            //begin mystery scaling numbers
            34: rom_data <= 16'h70_3a;
            35: rom_data <= 16'h71_35;
            36: rom_data <= 16'h72_11;
            37: rom_data <= 16'h73_f1;
            38: rom_data <= 16'ha2_02;
            //gamma curve values
            39: rom_data <= 16'h7a_20;
            40: rom_data <= 16'h7b_10;
            41: rom_data <= 16'h7c_1e;
            42: rom_data <= 16'h7d_35;
            43: rom_data <= 16'h7e_5a;
            44: rom_data <= 16'h7f_69;
            45: rom_data <= 16'h80_76;
            46: rom_data <= 16'h81_80;
            47: rom_data <= 16'h82_88;
            48: rom_data <= 16'h83_8f;
            49: rom_data <= 16'h84_96;
            50: rom_data <= 16'h85_a3;
            51: rom_data <= 16'h86_af;
            52: rom_data <= 16'h87_c4;
            53: rom_data <= 16'h88_d7;
            54: rom_data <= 16'h89_e8;
            //AGC and AEC
            55: rom_data <= 16'h13_e0;  //COM8, disable AGC / AEC
            56: rom_data <= 16'h00_00;  //set gain reg to 0 for AGC
            57: rom_data <= 16'h10_00;  //set ARCJ reg to 0
            58: rom_data <= 16'h0d_40;  //magic reserved bit for COM4
            59: rom_data <= 16'h14_18;  //COM9, 4x gain + magic bit
            60: rom_data <= 16'ha5_05;  // BD50MAX
            61: rom_data <= 16'hab_07;  //DB60MAX
            62: rom_data <= 16'h24_95;  //AGC upper limit
            63: rom_data <= 16'h25_33;  //AGC lower limit
            64: rom_data <= 16'h26_e3;  //AGC/AEC fast mode op region
            65: rom_data <= 16'h9f_78;  //HAECC1
            66: rom_data <= 16'ha0_68;  //HAECC2
            67: rom_data <= 16'ha1_03;  //magic
            68: rom_data <= 16'ha6_d8;  //HAECC3
            69: rom_data <= 16'ha7_d8;  //HAECC4
            70: rom_data <= 16'ha8_f0;  //HAECC5
            71: rom_data <= 16'ha9_90;  //HAECC6
            72: rom_data <= 16'haa_94;  //HAECC7
            73: rom_data <= 16'h13_e7;  //COM8, enable AGC / AEC
            74: rom_data <= 16'h69_07;
            default: rom_data <= 16'hFF_FF;  //mark end of ROM
        endcase
    end


endmodule
