import numpy as np


#dataset
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])



#initial weight and bias values 
b_init = 0
w_init = np.array([ 0.0, 0.0, 0.0, 0.0])



#define the cost function 
def costfunc(x,y,w,b):
    n=x.shape[0]
    summed=0.0
    for i in range(n):
        summed+=(np.dot(w,x[i])+b-y[i])**2
    summed=summed/2.0/n
    return summed

#computing the gradient
def comp_grad(x,y,w,b):
    m,n=x.shape
    djw=np.zeros((n))
    djb=0.0
    for i in range(m):
        for j in range(n): 
            djw[j]+=(np.dot(w,x[i])+b-y[i])*x[i,j]
        djb+=(np.dot(w,x[i])+b-y[i])
    djb=djb/m
    djw=djw/m

    return djw,djb

#gradient descent 
def graddesc(x,y,w,b,costfunction,gradientfunction,alpha,error,maxiterations):
    jinit=costfunction(x,y,w,b)
    wi=w
    bi=b
    for i in range(maxiterations):
        djw,djb=gradientfunction(x,y,w,b)
        wnew=w-alpha*djw
        bnew=b-alpha*djb
        jnew=costfunction(x,y,wnew,bnew)
        if (abs(jnew-jinit) < error):
            break
        else:
            jinit=jnew
            w=wnew
            b=bnew

    return wnew,bnew,costfunction(x,y,wnew,bnew)


#excecuting the gradient descnet linear regresison for out dataset
wfin,bfin,jfin=graddesc(x_train,y_train,inw,inb,costfunc,comp_grad,5.0e-7,1.0e-5,900)
print(wfin,bfin,jfin)
