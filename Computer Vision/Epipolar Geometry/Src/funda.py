from numpy import *

class ErrorShape(Exception):
    def __init__(self, x):
        self.x = x
        
    def __str__(self):
        return self.x
    
def fundamental_matrix(*args):
    try:
        x1,x2,npts = check_input(args)
        F = eight_point_algorithm(x1,x2)
        return F
    except ErrorShape, e:
        print 'ErrorShape, exception message:', e
        return None
    
def check_input(args):
    
    if len(args)==2:
        x1 = args[0]
        x2 = args[1]
        if not x1.shape == x2.shape:
            raise ErrorShape('the two arguments should have same size')
        
        if not x1.ndim == 2:
            raise ErrorShape('Each input has to be a 2D array')
        
        npts = max(x1.shape)
        d = min(x1.shape)
        if (npts<8):
            raise ErrorShape('At least 8 points are needed to compute the fundamental matrix')
        
        if d==2:
            if 2 == x1.shape[0]:  # input is a fat matrix
                x1 = r_[x1, ones((1,npts))]
                x2 = r_[x2, ones((1,npts))]
            else: # input is a tall matrix
                x1 = r_[x1.T, ones((1,npts))]
                x2 = r_[x2.T, ones((1,npts))]
        elif d==3:
            if 3 == x1.shape[1]:  # input is a tall matrix
                x1 = x1.T
                x2 = x2.T
        else:
            raise ErrorShape('x1 and x2 must be 2xN or Nx2 or 3xN or Nx3')

        
    elif len(args)==1:
        if not args[0].ndim == 2:
            raise ErrorShape('Each input has to be a 2D array')
        npts = max(args[0].shape)
        d = min(args[0].shape)
        if (npts<8):
            raise ErrorShape('At least 8 points are needed to compute the fundamental matrix')
        if d==4:
            if 4 == args[0].shape[0]:  # input is a fat matrix
                x1 = r_[args[0][0:2], ones((1,npts))]
                x2 = r_[args[0][2:4], ones((1,npts))]
            else: # input is a tall matrix
                x1 = r_[args[0][:,0:2].T, ones((1,npts))]
                x2 = r_[args[0][:,2:4].T, ones((1,npts))]
        elif d==6:
            if 6 == args[0].shape[0]:  # input is a tall matrix
                x1 = args[0][0:3]
                x2 = args[0][3:6]
            else:
                x1 = args[0][:,0:3].T
                x2 = args[0][:,3:6].T
        else:
            raise ErrorShape('Single argument x must be 4xN or Nx4 or 6xN or Nx6')
    else:
        raise ErrorShape('Wrong number of arguments supplied')
        
    return x1,x2,npts

    
def eight_point_algorithm(x1,x2):
    
    # perform the normalization
    x1, T1 = normalize2dpts(x1);
    x2, T2 = normalize2dpts(x2);
    A = constraint_matrix(x1,x2)
    (U, S, V) = linalg.svd(A)
    V = V.conj().T;
    F = V[:,8].reshape(3,3).copy()
    
    #F should be of rank 2
    (U,D,V) = linalg.svd(F);
    F = dot(dot(U,diag([D[0], D[1], 0])),V);

    #Denormalize
    F = dot(dot(T2.T,F),T1);
    return F
    
    
def constraint_matrix(x1,x2):
    npts = x1.shape[1]
    # stack column by column
    A = c_[x2[0]*x1[0], x2[0]*x1[1], x2[0], x2[1]*x1[0], x2[1]*x1[1], x2[1], x1[0], x1[1], ones((npts,1))]
    return A

def normalize2dpts(pts):
    if pts.shape[0]!=3:
        raise ErrorShape('points must be 3xN')
    
    finiteind = abs(pts[2]) > finfo(float).eps
    pts[0,finiteind] = pts[0,finiteind]/pts[2,finiteind]
    pts[1,finiteind] = pts[1,finiteind]/pts[2,finiteind]
    pts[2,finiteind] = 1
    
    # Centroid of finite points
    c = [mean(pts[0,finiteind]), mean(pts[1,finiteind])] 
    
    # Shift origin to centroid.
    newp0 = pts[0,finiteind]-c[0] 
    newp1 = pts[1,finiteind]-c[1] 

    meandist = mean(sqrt(newp0**2 + newp1**2));
    
    scale = sqrt(2)/meandist;
    T = eye(3)
    T[0][0] = scale
    T[1][1] = scale
    T[0][2] = -scale*c[0]
    T[1][2] = -scale*c[1]
    newpts = dot(T, pts)    
    
    return newpts, T