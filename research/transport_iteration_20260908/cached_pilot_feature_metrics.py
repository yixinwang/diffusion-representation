def measure(x,y):
 x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);n,m=len(x),len(y);d=x.shape[1]
 xx=x@x.T;yy=y@y.T;xy=x@y.T
 kxx=(xx/d+1)**3;kyy=(yy/d+1)**3;kxy=(xy/d+1)**3
 kid=(np.sum(kxx)-np.trace(kxx))/(n*(n-1))+(np.sum(kyy)-np.trace(kyy))/(m*(m-1))-2*np.mean(kxy)
 dx=np.sum(x*x,axis=1);dy=np.sum(y*y,axis=1)
 xx=np.maximum(dx[:,None]+dx[None,:]-2*xx,0);yy=np.maximum(dy[:,None]+dy[None,:]-2*yy,0);xy=np.maximum(dx[:,None]+dy[None,:]-2*xy,0)
 _,ix=np.unique(x,axis=0,return_inverse=True);_,iy=np.unique(y,axis=0,return_inverse=True)
 xx[ix[:,None]==ix[None,:]]=0;yy[iy[:,None]==iy[None,:]]=0
 _,joint=np.unique(np.concatenate((x,y)),axis=0,return_inverse=True)
 xy[joint[:n,None]==joint[None,n:]]=0
 np.fill_diagonal(xx,np.inf);np.fill_diagonal(yy,np.inf)
 rx=np.partition(xx,4,axis=1)[:,4];ry=np.partition(yy,4,axis=1)[:,4]
 inside_x=xy<rx[:,None];inside_y=xy<ry[None,:]
 return {'real_count':n,'fake_count':m,'kid_unbiased_full_bank':float(kid),
  'precision':float(inside_x.any(0).mean()),'recall':float(inside_y.any(1).mean()),
  'density':float(inside_x.sum(0).mean()/5),'coverage':float((xy.min(1)<rx).mean()),
  'nearest_k':5,'real_duplicate_rows':n-len(np.unique(ix)),'fake_duplicate_rows':m-len(np.unique(iy)),
  'cross_radius_ties_real':int((xy==rx[:,None]).sum()),'cross_radius_ties_fake':int((xy==ry[None,:]).sum())}
