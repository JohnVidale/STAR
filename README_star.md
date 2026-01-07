README

Main directory
plot_all_traces.py             plots all traces, one array per plot, 

all_traces_comp_shiftstack.py  plots all traces and compares shifted and unshifted stack for one component

stack_3comp.py                 make plot of the stack of 3 components on all arrays

show_1day.py   shows thumbnails of a day, clicking brings up full spectrograms

Compute statics
compute_shift - corrects for P or S arrival slowness, then estimates time shift for alignment by cc.
compute_statics - reads in shifts from compute shift, does sensible median to yield P and S statics

Spectro
data.mseed
Hao_spectro.py

Test_data
plot_making_shift.py
test_output.txt
TremorGUI.py