import torch

for i in range (1,20):
    if(i%10==0):
        torch.save({'test',i},'./test.txt')
        print("Saved!")