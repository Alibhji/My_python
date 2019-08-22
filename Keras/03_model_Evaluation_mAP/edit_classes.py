import  os
path='.\\input\\ground-truth'
path='.\\input\\detection-results'
# path='.\\input\\1'
path_o='.\\input\\out'
for counter,file in enumerate(os.listdir(path)):
    
    with open(os.path.join(path,file), 'r') as f:
        newTextFile=[]
        rows=f.readlines()
        for row in rows:
            k=row.split(' ')
            # if(len(k)==6): # this is for ground-truth
            if(len(k)==7):     # this is for detection-results      
                merge='_'.join(k[0:2])
                rest_of_row=' '.join(k[2:])
                newTextFile.append(merge+' '+rest_of_row)
            else:
                newTextFile.append(row)
                # print(merge)
                
    with open(os.path.join(path_o,file), 'w') as f:    
        for row in newTextFile:
            f.write('{}'.format(row))      
    print(counter,file)            

        
        
        
        
        #         obj=[]
        # rows=f.readlines()
        # for row in rows:
        #     # print(row , end='')
        #     if(len(row)>=2):
                
        #         if(row[0]+row[1]=="traffic light"):
        #             row[0]="traffic_light"
                    
        #         if(row[0]+row[1]=="fire hydrant"):
        #             row[0]="fire_hydrant"
                    
        #         if(row[0]+row[1]=="stop sign"):
        #             row[0]="stop_sign"
                    
        #         if(row[0]+row[1]=="parking meter"):
        #             row[0]="parking_meter" 
                    
        #         if(row[0]+row[1]=="baseball bat"):
        #             row[0]="baseball_bat"
                    
        #         if(row[0]+row[1]=="baseball glove"):
        #             row[0]="baseball_glove"
                    
        #         if(row[0]+row[1]=="tennis racket"):
        #             row[0]="tennis_racket"
                    
        #         if(row[0]+row[1]=="wine glass"):
        #             row[0]="wine_glass"  
                    
        #         if(row[0]+row[1]=="hot dog"):
        #             row[0]="hot_dog"  
                    
        #         if(row[0]+row[1]=="cell phone"):
        #             row[0]="cell_phone"   
                    
        #         if(row[0]+row[1]=="teddy bear"):
        #             row[0]="teddy_bear"  
                    
        #         if(row[0]+row[1]=="hair drier"):
        #             row[0]="hair_drier"  
            
        #     label = "{} {}".format(row[0], row[1:-1])
        #     obj.append(label)
            
        # print('-----------------------')
        # print(obj)