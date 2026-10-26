import sys, numpy as np
from PIL import Image
from scipy import ndimage
src,out,maxs=sys.argv[1],sys.argv[2],int(sys.argv[3])
im=np.asarray(Image.open(src).convert('RGB')).astype(np.float32)
r,g,b=im[...,0],im[...,1],im[...,2]
border=np.concatenate([im[:6].reshape(-1,3),im[-6:].reshape(-1,3),im[:,:6].reshape(-1,3),im[:,-6:].reshape(-1,3)])
bg=np.median(border,axis=0);mx=np.maximum(r,b);dom=g-mx;dist=np.linalg.norm(im-bg,axis=2)
cand=(dom>40)&(dist<170)
lab,n=ndimage.label(cand)
edge=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])))-{0}
bgm=np.isin(lab,list(edge))
if n:
    sizes=ndimage.sum(np.ones_like(dist),lab,range(1,n+1));md=ndimage.mean(dist,lab,range(1,n+1))
    for i in range(n):
        if (i+1) not in edge and sizes[i]>120 and md[i]<75: bgm|=(lab==i+1)
near=ndimage.binary_dilation(bgm,iterations=3)
a=np.where(bgm,0.0,1.0);soft=np.clip((75-dom)/55,0,1);a=np.where(near&~bgm,np.minimum(a,soft),a);a=ndimage.gaussian_filter(a,0.7)
im[...,1]=np.where(near|(dom>15),np.minimum(g,mx+np.clip(dom,0,30)*0.2),g)
rgba=np.dstack([im,a*255]).clip(0,255).astype(np.uint8)
ys,xs=np.nonzero(rgba[...,3]>10);rgba=rgba[ys.min():ys.max()+1,xs.min():xs.max()+1]
o=Image.fromarray(rgba,'RGBA');o.thumbnail((maxs,maxs),Image.LANCZOS);o.save(out,'WEBP',quality=86,method=6);print(out,o.size)
