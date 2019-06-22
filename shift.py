import numpy as np
def shift(x, k, l, boundary):
    if x.ndim == 2:
        color = 1
    else :
        color = 3
    n1 = np.shape(x)[0]
    n2 = np.shape(x)[1]
    xshifted = np.zeros((n1,n2,color))
    irange = np.mod(np.arange(n1) + k, n1)
    jrange = np.mod(np.arange(n2) + l, n2)
    # firstly move upward then move rightward
    xshifted = x[irange, :][:, jrange]
    if boundary == 'periodical':
        pass
    elif boundary is 'extension':
        m = n1 - k if k > 0 else -k-1
        n = n2 - l if l > 0 else -l-1
        if k != 0:
            xshifted[m::np.sign(k),:,:] = np.tile(xshifted[m-np.sign(k):m-np.sign(k)+1,:,:],(np.sign(k)*k,1,1))
        if l != 0:
            xshifted[:,n::np.sign(l),:] = np.tile(xshifted[:,n-np.sign(l):n-np.sign(l)+1,:],(1,np.sign(l)*l,1))
    elif boundary == 'zero-padding':
        period = xshifted
        xshifted = np.zeros_like(period)
        m = n1 - k if k > 0 else -k-1
        n = n2 - l if l > 0 else -l-1  
        sign_k = np.sign(k) if k != 0 else 1 
        sign_l = np.sign(l) if l != 0 else 1
        if k == 0:
            m = n1
        if l == 0:
            n = n2
        xshifted[:m:sign_k,:n:sign_l,:] = period[:m:sign_k,:n:sign_l,:]
    # mirror
    else:
        m = n1 - k if k > 0 else -k
        n = n2 - l if l > 0 else -l
        add_k = 1 if k < 0 else 0
        add_l = 1 if l < 0 else 0
        if k != 0:
            xshifted[m::np.sign(k),:,:] = xshifted[min(m,m-k):max(m,m-k) + add_k,:,:][::-np.sign(k),:,:]
        if l != 0:
            xshifted[:,n::np.sign(l),:] = xshifted[:,min(n,n-l):max(n,n-l) + add_l ,:][:,::-np.sign(l),:]
    return xshifted
