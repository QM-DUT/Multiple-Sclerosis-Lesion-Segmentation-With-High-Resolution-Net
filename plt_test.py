import matplotlib.pyplot as plt
import numpy as np
a=np.zeros((5,10,11))
b=a.transpose(1,2,0)
b=np.pad(b, ([1,1], [1,1], [1,1]), 'constant')
print(b.shape)
c=b.transpose(2,0,1)
print(c.shape)





