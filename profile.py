import numpy as np
import time

data = np.zeros([100,20,224,224])
l = 10
h = 256
w = 340
c = 2

I = np.random.rand(256,340,2)
start = time.time()
for x in xrange(10):
    D = []
    for i in xrange(10):
        d = np.copy(I)
        if i == 0:
            D = d
        else:
            D = np.concatenate((D,d), axis=2)
    DD = D[:,::-1,:]
    D = np.concatenate((D[:,:,:,np.newaxis],D[:,:,:,np.newaxis]), axis=3)
    d = np.transpose(D, (3,2,0,1))
    print d.shape
    crop = np.concatenate((d[:,:,0:224,0:224],\
                    d[:,:,h-224:h,0:224],\
                    d[:,:,0:224,w-224:w],\
                    d[:,:,h-224:h,w-224:w],\
                    d[:,:,0:224,0:224]), axis=0)
    data[x*l:(x+1)*l,:,:,:] = crop
end = time.time()
print "%.2f" % (end-start)
