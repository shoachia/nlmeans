import numpy as np
import numpy.fft as npf
import matplotlib
import matplotlib.pyplot as plt
import time
import imagetools as im


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
        if color == 3:
            xshifted[:m:sign_k,:n:sign_l,:] = period[:m:sign_k,:n:sign_l,:]
        else:
            xshifted[:m:sign_k,:n:sign_l] = period[:m:sign_k,:n:sign_l]
    # mirror
    else:
        m = n1 - k if k > 0 else -k
        n = n2 - l if l > 0 else -l
        add_k = 1 if k < 0 else 0
        add_l = 1 if l < 0 else 0
        if color == 3:
            if k != 0:
                xshifted[m::np.sign(k),:,:] = xshifted[min(m,m-k):max(m,m-k) + add_k,:,:][::-np.sign(k),:,:]
            if l != 0:
                xshifted[:,n::np.sign(l),:] = xshifted[:,min(n,n-l):max(n,n-l) + add_l ,:][:,::-np.sign(l),:]
        else:
            if k != 0:
                xshifted[m::np.sign(k),:] = xshifted[min(m,m-k):max(m,m-k) + add_k,:][::-np.sign(k),:]
            if l != 0:
                xshifted[:,n::np.sign(l)] = xshifted[:,min(n,n-l):max(n,n-l) + add_l][:,::-np.sign(l)]
            
    return xshifted


def kernel(name, tau=1, eps=1e-3):
    if name == 'gaussian':
        s1 = 0
        while True:
            if np.exp(-(s1**2)/(2*tau)) < eps:
                break
            s1 += 1
        s1 = s1-1
        s2 = s1
        i = np.arange(-s1,s1+1) #-3 ~ 3
        j = np.arange(-s2,s2+1) #-3 ~ 3 
        ii, jj = np.meshgrid(i, j, sparse=True,indexing='ij')
        nu = np.exp(-(ii**2 + jj**2) / (2*tau**2))
        nu[nu < eps] = 0
        nu /= nu.sum()
    elif name == 'exponential':
        if tau == 0:
            s1 = 1
            s2 =1
        else:
            s1 = 0
            while True:
                if np.exp(-(s1)/(tau)) < eps:
                    break
                s1 += 1
            s1 = s1-1
            s2 = s1
            
        i = np.arange(-s1,s1+1) #-20 ~ 20
        j = np.arange(-s2,s2+1) #-20 ~ 20
        ii, jj = np.meshgrid(i, j, sparse=True,indexing='ij')
        if tau == 0: 
            tau = 1e-3
        nu = np.exp(-(np.sqrt(ii**2+jj**2))/tau)
        nu[nu < eps] = 0
        nu /= nu.sum()
    elif name.startswith('box'):
        if name.endswith('1'):
            nu = np.zeros((2*tau+1, 1))
            nu[:,0] = 1/(2*tau+1)
        elif name.endswith('2'):
            nu = np.zeros((1, 2*tau+1))
            nu[0,:] = 1/(2*tau+1)
        else:
            s1 = tau
            s2 = tau
            i = np.arange(-s1,s1+1) #-1 ~ 1
            j = np.arange(-s2,s2+1) #-1 ~ 1
            ii, jj = np.meshgrid(i, j, sparse=True,indexing='ij')
            nu = np.exp(0*(ii+jj))
            nu[nu < eps] = 0
            nu /= nu.sum()
    elif name == 'motion':
        nu = np.load('./assets/motionblur.npy')
    elif name.endswith('forward'):
        if name.startswith('1',4,5):
            nu = np.zeros((3, 1))
            nu[1, 0] = 1
            nu[2, 0] = -1
        elif name.startswith('2',4,5):
            nu = np.zeros((1, 3))
            nu[0, 1] = 1
            nu[0, 2] = -1
        else:
            raise ValueError('invalid kernel')
    elif name.endswith('backward'):
        if name.startswith('1',4,5):
            nu = np.zeros((3, 1))
            nu[0, 0] = 1
            nu[1, 0] = -1
        elif name.startswith('2',4,5):
            nu = np.zeros((1, 3))
            nu[0, 0] = 1
            nu[0, 1] = -1
        else: 
            raise ValueError('invalid kernel')
    elif name.startswith('laplacian'):
        if name.endswith('1'):
            nu = np.zeros((3, 1))
            nu[0, 0] = 1
            nu[1, 0] = -2
            nu[2, 0] = 1 
        elif name.endswith('2'):
            nu = np.zeros((1, 3))
            nu[0, 0] = 1
            nu[0, 1] = -2
            nu[0, 2] = 1
        else: 
            raise ValueError('invalid kernel')
    else:
        raise ValueError('invalid kernel')
    return nu

def convolve_naive(x, nu):
    n1, n2 = x.shape[:2]
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    xconv = np.zeros(x.shape)
    for i in range(s1, n1-s1):
        for j in range(s2, n2-s2):
            #kernel part
            for k in range(-s1, s1+1):
                for l in range(-s2, s2+1):
                    xconv[i][j] += x[i-k][j-l]*nu[k+s1][l+s2]
    return xconv

def convolve(x, nu, boundary ):
    xconv = np.zeros(x.shape)
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            xconv += nu[k+s1,l+s2]*shift(x,-k,-l,boundary)
    return xconv

def kernel2fft(nu, n1, n2, separable = None):
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    tmp = np.zeros((n1,n2))
    tmp[:s1+1, :s2+1] = nu[s1:2*s1+1, s2:2*s2+1]
    tmp[n1-s1:n1, :s2+1] = nu[:s1,s2:2*s2+1]
    tmp[n1-s1:n1,n2-s2:n2] = nu[:s1,:s2]
    tmp[:s1+1,n2-s2:n2] = nu[s1:2*s1+1,:s2]
    lbd = npf.fft2(tmp)
    return lbd

def convolvefft(x, lbd):
    return np.real(npf.ifftn(npf.fftn(x)*lbd[:,:,None]))

def nu2mu(nu):
    return np.fliplr(np.flipud(nu))

def phi(alpha,C,h,sig):
    a = np.maximum(alpha-2*h*(sig**2),0)
    b = 2*np.sqrt(2)*h*(sig**2)/np.sqrt(C)
    inner = -a/b
    return np.exp(inner)
def bilateral_naive(y, sig, s1=2, s2=2, h=1):
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    for i in range(s1, n1-s1):
        for j in range(s2, n2-s2):
            tmp1 = 0
            tmp2 = 0
            for k in range(-s1, s1 + 1):
                for l in range(-s2, s2 + 1):
                    dist2 = ((y[i + k, j + l] - y[i, j])**2).mean()
                    tmp1 += phi(dist2,c,h,sig) * y[i + k][j + l]   
                    tmp2 += phi(dist2,c,h,sig)
            x[i][j] = tmp1
            Z[i][j] = tmp2 
    Z[Z == 0] = 1
    x = x / Z
    return x
def bilateral(y, sig, s1=10, s2=10, h=1, boundary= 'mirror'):
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    for k in range(-s1, s1 + 1):
        for l in range(-s2, s2 + 1):
            tmp = phi(((shift(y,k,l,boundary) - y)**2).mean(axis = 2),c,h,sig)
            tmp_new = np.moveaxis((tmp.reshape(1,n1,n2)),[0,1,2],[2,0,1])
            x += tmp_new*shift(y,k,l,boundary)
            Z += tmp_new
    Z[Z == 0] = 1
    x = x / Z
    return x
def phi_nl(alpha,C,h,sig,P):
    a = np.maximum(alpha-2*h*(sig**2),0)
    b = 2*np.sqrt(2)*h*(sig**2)/np.sqrt(C*P)
    inner = -a/b
    return np.exp(inner)
def nlmeans_naive(y, sig, s1=2, s2=2, p1=1, p2=1, h=1):
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim-2)))
    
    P = (2*p1+1)*(2*p2+1)
    for i in range(s1, n1-s1-p1):
        for j in range(s2, n2-s2-p2):
            for k in range(-s1, s1+1):
                for l in range(-s2, s2+1):
                    dist2 = 0
                    for u in range(-p1, p1+1):
                        for v in range(-p2, p2+1):
                            dist2 += ((y[i+k+u, j+l+v]-y[i+u, j+v])**2)
                    kernel = phi_nl(dist2.mean()/P, c,h,sig, P)
                    Z[i,j] += kernel
                    x[i,j] += kernel*y[i+k,j+l]
                    
    Z[Z==0] = 1
    x = x/ Z
    return x
def nlmeans(y, sig, s1=7, s2=7, p1=None, p2=None, h=1, boundary='mirror'):
    p1 = (1 if y.ndim == 3 else 2) if p1 is None else p1
    p2 = (1 if y.ndim == 3 else 2) if p2 is None else p2
    # Making box kernel
    P = (2*p1+1)*(2*p2+1) 
    i = np.arange(-p1,p1+1) #-1 ~ 1
    j = np.arange(-p2,p2+1) #-1 ~ 1
    ii, jj = np.meshgrid(i, j, sparse=True,indexing='ij')
    nu = np.exp(0*(ii+jj))
    nu[nu < 1e-3] = 0
    nu /= nu.sum()
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    #only need two loop --> there are also two loops in convolve function
    for k in range(-s1, s1 + 1):
        for l in range(-s2, s2 + 1):
            #using convolve function
            tmp = phi_nl((convolve((shift(y,k,l,boundary) - y)**2,nu,boundary)).mean(axis = 2),c,h,sig,P)
            x += tmp[:,:,None]*shift(y,k,l,boundary)
            Z += tmp[:,:,None]
    x = x / Z
    return x
def psnr(x,x0):
    n1, n2 = x.shape[:2]
    c = x.shape[2] if x.ndim == 3 else 1
    return 10*np.log10(n1*n2*c/(np.linalg.norm(x-x0)**2))
def convolve_sep(x, nu, boundary='periodical', separable=None):
    if separable == 'product':
        tmp = convolve(x,nu[0],boundary)
        xconv = convolve(tmp,nu[1],boundary)
    # sum is only for laplacian ????
    elif separable == 'sum':
        tmp1 = convolve(x,nu[0],boundary)
        tmp2 = convolve(x,nu[1],boundary)
        xconv = tmp1 + tmp2
    else: 
        xconv = convolve(x,nu,boundary)
    return xconv
def laplacian(x, boundary = 'periodical'):
    #Using separable == 'sum'
    nu1 = kernel('laplacian1')
    nu2 = kernel('laplacian2')
    nu = (nu1,nu2)
    xconv = convolve_sep(x, nu, boundary='periodical', separable='sum')
    return xconv
def grad(x, boundary='periodical'):
    g = np.stack((convolve_sep(x,kernel('grad1_forward'),boundary),convolve_sep(x,kernel('grad2_forward'),boundary)),axis = 2)
    return g
def div(f, boundary='periodical'):
    # f is a nxnx2x3 array for RGB image
    # grad1_forward will get the wrong answer
    d = convolve_sep(f[:,:,0],kernel('grad1_backward')) + convolve_sep(f[:,:,1],kernel('grad2_backward'))
    return d
def average_power_spectral_density(x):
    K = len(x)
    tmp = np.zeros(x[0].shape[:2])
    
    for i in range(K):
        tmp += np.absolute(npf.fft2(x[i].mean(axis = 2)))**2
    
    S = (tmp / K)
    
    return S
def mean_power_spectrum_density(apsd):
    
    n1,n2 = apsd.shape
    s = np.log(apsd)- np.log(n1) - np.log(n2)
    vv,uu = im.fftgrid(n1,n2)
    w = np.sqrt((vv/n1)**2 + (uu/n2)**2)
    
    w_tmp = w.flatten()
    w_new = w_tmp[1:]
    t = np.log(w_new)
    s_tmp = s.flatten()
    s_new = s_tmp[1:]
    
    alpha = np.sum((t-t.mean())*(s_new-s_new.mean()))/(np.sum((t-t.mean())**2))
    beta = s_new.mean() - alpha * t.mean()
    
    #alpha  = ((t*s_new).sum()- n1*n2*s_new.mean()*t.mean())/((t**2).sum()- n1*n2*(t.mean())**2)
    #beta = s_new.mean() - alpha*t.mean()
    mpsd = (n1-1)*(n2-1)*np.exp(beta)*(w**(alpha))
    mpsd[0,0] = np.inf
    return mpsd, alpha, beta

def deconvolve_naive(y, lbd, return_transfer=False):
    y_hat = npf.fftn(y)
    hhat = np.conjugate(lbd)/(np.abs(lbd)**2)
    xdec = np.real(npf.ifftn(hhat[:,:,None]*y_hat))
    if return_transfer:
        return xdec, hhat
    else:
        return xdec
def deconvolve_wiener(x, lbd, sig, mpsd, return_transfer=False):
    n1,n2 = x.shape[:2]
    x_hat = npf.fftn(x)
    hhat = np.conjugate(lbd)/(np.abs(lbd)**2 + n1*n2*(sig**2)/mpsd)    
    xdec = np.real(npf.ifftn(hhat[:,:,None]*x_hat))

    if return_transfer:
        return xdec, hhat
    else:
        return xdec
