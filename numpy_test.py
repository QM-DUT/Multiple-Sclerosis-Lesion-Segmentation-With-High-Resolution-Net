import numpy as np
from utils.util import *





a=np.ones((16,4,64,64,64))

b=np.ones((3,4,2,2,2))

a[[1,5,10],:,2:4,4:6,6:8]+=b



a=[3,2,1.6,8]
ii=sorted(range(len(a)),key=lambda k:a[k])
print(ii)



a=np.ones((181,217,181))
a=a[10:170,12:204,10:170]
print(a.shape)


a=np.random.rand(2,2,2,2)
b=a.transpose(0, 2, 3, 1)
c=b.transpose(0, 3, 1, 2)
print(a)
print(222222)
print(c)

# for root, dirs, files in os.walk("../ISBI2015/"):
#     print(root, dirs, files)
# if __name__=="__main__":
# load_data_full_4("../ISBI2015/",1)
#     # for root, dirs, files in os.walk("../ISBI2015/"):
#     #     print(root, dirs, files)
