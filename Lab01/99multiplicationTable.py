#!/usr/bin/env python
# coding: utf-8

# In[5]:
# i是被乘數
for i in range(1,10):
    #j是乘數
    for j in range(1,10):
        #把所有結果輸出
        print("{} * {} = {}".format(i,j,i*j))
    #每一部分換行
    print("")