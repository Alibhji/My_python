
import time

def printProgressBar(iteration,total,prefix='',suffix='',decimals=1,length=100,fill='█'): # alt+219
    percent=("{0:."+str(decimals)+"f}").format(100*(iteration/float(total)))
    filledlength=int(length*iteration//total)
    bar=fill*filledlength+'-'*(length-filledlength)
    print('\r%s|%s| %s%% (%s/%s)  %s' % (prefix,bar,percent,iteration,total,suffix),end='\r')
    if iteration == total:
        print("\n")



for i in range(100):
    progress_len=100
    progress_cnt=i+1
    printProgressBar(progress_cnt,progress_len,prefix='Ali Test 1:'.ljust(15),suffix='Compllete',length=50)
    time.sleep(0.01)

