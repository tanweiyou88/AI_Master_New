import numpy as np

a=np.array([1,2,3])
b=np.array([4,5,6])
c=np.array([0])
d={'k':0, 'i':1, 'j':2}
e=[7,8,9]

def add1():
    c=a+b
    return c

def add2(c):
    c=a+b

def add2a(c):
    c=a+b
    return c

def add3():
    c=a+b

def add4():
    return a+b

def add5(c,d):
    return c+d

def funct1():
    d["k"]=5

def funct2():
    b=np.array([11,12,13])

def funct2b():
    b=np.array([11,12,13])
    return b

def funct3():
    e=[22,33,44]

def funct4():
    e=[22,33,44]
    return e

def funct5(e):
    e=[22,33,44]
    return e

def funct6():
    e.append([22,33,44])
    return e



print("Initial values:\n")
print("a:",a)
print("b:",b)
print("c:",c)
print("e:",e)
print("d['k']: ",d['k'])
print("d['i']: ",d['i'])
print("d['j']: ",d['j'])


answer1 = add1()
print("answer1: ",answer1)
answer2 = add2(c)
print("answer2: ",answer2)
answer2a = add2a(c)
print("answer2a: ",answer2a)
answer3 = add3()
print("answer3: ",answer3)
answer4 = add4()
print("answer4: ",answer4)
answer5 = add5(a,b)
print("answer5: ",answer5)
funct1()
print("d['k']: ",d['k'])
funct2()
print("b: ",b)
r = funct2b()
print("r: ",r)
print("b: ",b)
funct3()
print("e: ",e)
funct4()
print("e: ",e)
funct5(e)
print("e: ",e)
funct6()
print("e: ",e)

print(b[:3])
print(b[1:])
