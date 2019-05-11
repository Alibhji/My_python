






def printProgressBar(iteration,total,prefix='',suffix='',decimals=1,length=100,fill='█'): # alt+219
    percent=("{0:."+str(decimals)+"f}").format(100*(iteration/float(total)))
    filledlength=int(length*iteration//total)
    bar=fill*filledlength+'-'*(length-filledlength)
    print('\r%s|%s| %s%% (%s/%s)  %s' % (prefix,bar,percent,iteration,total,suffix),end='\r')
    if iteration == total:
        print("\n")


class VOC:
    def xml_indent(self,elem,level=0):
        i="\n"+level*"\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text=i+"\t"