# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:07:15 2018

@author: huangjianxiang huangjianxiang@zju.edu.cn
"""
import numpy as np
from math import ceil
from timeit import default_timer as timer
from io import StringIO
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import gc
from collections import deque
from numba import jit
from array import array

class ESP_extract:
    def __init__(self,filename):
        self.filename=filename
        self.n=int(self.extract_head()[0][0])
        self.a=int(self.extract_head()[1][0])
        self.b=int(self.extract_head()[2][0])
        self.c=int(self.extract_head()[3][0])
        
        self.x0=self.extract_head()[0][1]
        self.y0=self.extract_head()[0][2]
        self.z0=self.extract_head()[0][3]
        
        self.xstep=self.extract_head()[1][1]
        self.ystep=self.extract_head()[2][2]
        self.zstep=self.extract_head()[3][3]
        

        print("Number of atoms:",self.n)
        print("Number of bins along X:",self.a)
        print("Number of bins along Y:",self.b)
        print("Number of bins along Z:",self.c)

        print("step along X:",self.xstep)
        print("step along Y:",self.ystep)
        print("step along Z:",self.zstep)

    def extract_head(self):
        arr_in=np.zeros((4,4))
        file1=open(self.filename,"r")
        file1.readline()
        file1.readline()

        for i in range(0,4):
            a=file1.readline()
            arr_in[i][0]=float(a[0:5])
            arr_in[i][1]=float(a[5:17])
            arr_in[i][2]=float(a[17:29])
            arr_in[i][3]=float(a[29:41])
        file1.close()
        #print(arr_in)
        return arr_in
    
    def extract_atom(self):
        # n is the number of atoms
        n=self.n
        arr_in = np.zeros((n,5))
        file1=open(self.filename,"r")
        for i in range(0,6):
            file1.readline()
        for i in range(0,n):
            a=file1.readline()
            arr_in[i][0]=float(a[0:5])
            arr_in[i][1]=float(a[5:17])
            arr_in[i][2]=float(a[17:29])
            arr_in[i][3]=float(a[29:41])
            arr_in[i][4]=float(a[41:53])
        file1.close()
        return arr_in

    def extract_den(self):
        a=self.a
        b=self.b
        c=self.c
        n=self.n
        
        x0=self.x0
        y0=self.y0
        z0=self.z0
        
        xstep=self.xstep
        ystep=self.ystep
        zstep=self.zstep
        
        filename=self.filename
        
        # arr3 is the cubic electron densities
        arr4=np.zeros((a,b,c))
        # arr5=np.zeros((a,b,c)) #mark all the grid data with label 0 or 1
        
        
        #extract electron densities
        file1=open(filename,"r")
        
        for i in range(0,n+6):
            file1.readline()
        newcontent=(file1.read()).replace("\n", "")
        np_grid=np.genfromtxt(StringIO(newcontent))
        arr3=np_grid.reshape((a,b,c))
        file1.close()
        
        #extract electrostatic potential--->extract data
        filename1=self.filename.split(".")[0]+".pot"
        file3=open(filename1,"r")
        
        for i in range(0,n+6):
            file3.readline()
        newcontent=(file3.read()).replace("\n", "")
        np_grid=np.genfromtxt(StringIO(newcontent))
        arr6=np_grid.reshape((a,b,c))
        file3.close()
        
        arr4=(arr3<=0.001)# True for exterior; False for interior
        
        cube_count=[]
        # coord=np.zeros(5)
        for i in range(0,a-1):  
            for j in range(0,b-1):  
                for k in range(0,c-1):
                    if (arr4[i,j,k]==arr4[i+1,j,k]==arr4[i,j+1,k]==arr4[i,j,k+1]==arr4[i+1,j+1,k]==arr4[i+1,j,k+1]==arr4[i,j+1,k+1]==arr4[i+1,j+1,k+1]==0) or \
                        (arr4[i,j,k]==arr4[i+1,j,k]==arr4[i,j+1,k]==arr4[i,j,k+1]==arr4[i+1,j+1,k]==arr4[i+1,j,k+1]==arr4[i,j+1,k+1]==arr4[i+1,j+1,k+1]==1):
                            pass
                    else:
                        # arr5[i,j,k]=1
                        # arr5[i+1,j,k]=1
                        # arr5[i,j+1,k]=1
                        # arr5[i,j,k+1]=1
                        # arr5[i+1,j+1,k]=1
                        # arr5[i+1,j,k+1]=1
                        # arr5[i,j+1,k+1]=1
                        # arr5[i+1,j+1,k+1]=1
                        
                        coordx=i*xstep+x0
                        coordy=j*ystep+y0
                        coordz=k*zstep+z0   
                        coordx1=(i+1)*xstep+x0
                        coordy1=(j+1)*ystep+y0
                        coordz1=(k+1)*zstep+z0   

                        cube_count.append([coordx,coordy,coordz,arr3[i,j,k],arr6[i,j,k]])                        
                        cube_count.append([coordx1,coordy,coordz,arr3[i+1,j,k],arr6[i+1,j,k]])
                        cube_count.append([coordx,coordy1,coordz,arr3[i,j+1,k],arr6[i,j+1,k]])
                        cube_count.append([coordx,coordy,coordz1,arr3[i,j,k+1],arr6[i,j,k+1]])
                        cube_count.append([coordx1,coordy1,coordz,arr3[i+1,j+1,k],arr6[i+1,j+1,k]])
                        cube_count.append([coordx1,coordy,coordz1,arr3[i+1,j,k+1],arr6[i+1,j,k+1]])
                        cube_count.append([coordx,coordy1,coordz1,arr3[i,j+1,k+1],arr6[i,j+1,k+1]])
                        cube_count.append([coordx1,coordy1,coordz1,arr3[i+1,j+1,k+1],arr6[i+1,j+1,k+1]])
                        
        #coord is set to has one dimension of 5
        #the first three coords are x y z,respectively
        #the fourth is the electron density
        #the last is electrostatic potentials
                        

        
        #extract electron densities and electrostatic potentials
        """
        coord=np.zeros(5)
        for i in range(a):  
            for j in range(b):  
                for k in range(c):
                    if arr5[i,j,k]:                        
                        coordx=i*xstep+x0
                        coordy=j*ystep+y0
                        coordz=k*zstep+z0                        
                        coord=np.append(coord,np.array([coordx,coordy,coordz,arr3[i,j,k],arr6[i,j,k]]))
        """
        
        #coord=coord.reshape(np.size(coord)//5,5)
        #num=np.shape(coord)[0]
        
        num=len(cube_count)
        
        coord=np.array(cube_count)
        file2=open("grid.xyz","w")
        content=""
        for l in range(num):
            if l>=1:
                # print(coord[l,:])
                content+=("C         "+str(coord[l,0])+" "+str(coord[l,1])+" "+str(coord[l,2])+"\n")

        file2.write(str(num-1)+"\n\n")
        file2.write(content)
        file2.close()
        # # Creating figure
        # fig = plt.figure(figsize = (10, 10))
        # ax = plt.axes(projection ="3d")
        # ax.scatter3D(coord[:,0], coord[:,1], coord[:,2], color = "green")
        # plt.savefig("test.png",dpi=300)
        #return coorinates of the surface and the markers of the surface grids
        
        
        coord
        # return coord,arr5,cube_count
        return coord, cube_count

#MT algorithm|8|8|8|8|8|8|....


def marching_tetra(input):
    conect=[]
    # test=np.array(cube_count)
    test=np.array(input)
    # gc.disable()
    # new_coord=deque([])
    # new_coord=array("f")
    new_coord=[]
    # id=0
    cum=np.array([0,0,0,0,0])
    cum=cum.reshape(1,5)
    
    for i in range(np.shape(test)[0]//8):
        block=test[i*8:i*8+8,:]
        # print(i)
        # print(np.shape(new_coord))
        # print(np.shape(cum))

        if i%1000==0:
            print(i)
            print(timer())
            if i>0:
                cum=np.vstack((cum,np.array(new_coord)))
                # print(np.shape(cum))
                # print(np.shape(new_coord))
                new_coord=[]
            
            
        
        #tetra=[[2,1,0,3],[2,1,5,3],[2,6,5,3],[2,5,4,1],[2,5,4,6],[7,5,4,6]]
        
        tetra=[[5, 2, 4, 1],[5, 2, 4, 7],[5, 2, 6, 7],[5, 2 ,6 ,3],[5, 2, 0, 3],[5, 2 ,0 ,1]]
        
        
        
        for j in range(6):
            block_cut=block[tetra[j],:]
            #count=0
            for k in range(6):
                count=0	
                pair_list=[]
                if (block_cut[0,3]-0.001)*(block_cut[1,3]-0.001)<0:
                    count+=1
                    pair_list.append([0,1])
                if (block_cut[0,3]-0.001)*(block_cut[2,3]-0.001)<0:
                    count+=1
                    pair_list.append([0,2])
                if (block_cut[0,3]-0.001)*(block_cut[3,3]-0.001)<0:
                    count+=1
                    pair_list.append([0,3])
                if (block_cut[1,3]-0.001)*(block_cut[2,3]-0.001)<0:
                    count+=1
                    pair_list.append([1,2])
                if (block_cut[1,3]-0.001)*(block_cut[3,3]-0.001)<0:
                    count+=1
                    pair_list.append([1,3])
                if (block_cut[2,3]-0.001)*(block_cut[3,3]-0.001)<0:
                    count+=1
                    pair_list.append([2,3])
                # print(pair_list)        
                if count==3:
                    for l in range(3):
                        a,b=pair_list[l]
                        one=block_cut[a,3]
                        two=block_cut[b,3]
                        
                        # print(a,b)
                        xishu=(one-0.001)/(one-two)
                        # print(xishu)
                        new=(1-xishu)*block_cut[a,:]+xishu*block_cut[b,:]
                        # print(new)
                        new=new.tolist()
                        if new not in new_coord:
                            new_coord.append(new)

                            # cum=np.vstack((cum,np.array(new_coord)))
                            # print(np.shape(cum))
                            # new_coord=[]
                            # id+=1
                        # new_coord=np.append(new_coord,new)
                elif count==4:
                    for l in range(4):
                        a,b=pair_list[l]
                        one=block_cut[a,3]
                        two=block_cut[b,3]
                        # print(a,b)
                        xishu=(one-0.001)/(one-two)
                        # print(xishu)
                        new=(1-xishu)*block_cut[a,:]+xishu*block_cut[b,:]
                        # print(new)
                        new=new.tolist()
                        if new not in new_coord:
                            new_coord.append(new)
                            # cum=np.vstack((cum,np.array(new_coord)))
                            # print(np.shape(cum))
                            # new_coord=[]
                            # id+=1
                        # new_coord=np.append(new_coord,new) 
    # gc.enable()
    #cum=np.vstack((cum,np.array(new_coord)))
    return cum



if __name__ == '__main__':
    print(timer())
    x=ESP_extract("C53.den")
    
    coord,cube_count=x.extract_den()
    print(timer())
    
    cube_count=np.array(cube_count)
    output=marching_tetra(cube_count)
    

    #test code
    # test=np.array([[0,0,0,0.0008,1],[1,0,0,0.0008,1],[0,1,0,0.0008,1],[0,0,1,0.0008,1],[1,1,0,0.0012,1],[1,0,1,0.0008,1],[0,1,1,0.0008,1],[1,1,1,0.0008,1]])
    
    # output=marching_tetra(test)
    
    leng=np.size(output)//5
    output=np.array(output)
    output= output.reshape(leng,5)
    num=np.shape(output)[0]
    
    file2=open("grid_cut.xyz","w")
    content=""
    for l in range(num):
        if l>=1:
            # print(coord[l,:])
            content+=("C         "+str(output[l,0])+" "+str(output[l,1])+" "+str(output[l,2])+"\n")
    file2.write(str(num-1)+"\n\n")
    file2.write(content)
    file2.close()
    
    # e = timer()

    # vmin=arr3.min()
    # vsmax=arr5.max()
    # vsmin=arr5.min()
    # vsmean=arr5.mean()
    # print ("vmin is ",vmin)
    # print ("vsmax is ",vsmax)
    # print ("vsmin is",vsmin)
    # print ("vsmean is",vsmean)
    # print ("\nThe time spent was " +'{:.1f}'.format(e - s)+" seconds!")













