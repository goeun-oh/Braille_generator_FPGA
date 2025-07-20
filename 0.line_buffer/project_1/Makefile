# Makefile
# 파일명 설정
TOP_MODULE     = tb_line_buffer
SV_FILES       = line_buffer.v tb_line_buffer.sv
WORK_DIR       = work
WDB_FILE       = sim.wdb
LOG_FILE       = sim.log

all:
	xvlog --sv $(SV_FILES) > compile.log || (cat compile.log && exit 1)

	xelab $(TOP_MODULE) --debug typical -log elaborate.log -L xil_defaultlib -s $(TOP_MODULE)_elab

	xsim $(TOP_MODULE)_elab -t xsim.tcl > $(LOG_FILE)

clean:
	rm -f *.jou *.log
