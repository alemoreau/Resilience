from scipy.sparse.linalg.isolve import _iterative

from scipy.sparse.linalg.interface import LinearOperator
from scipy._lib.decorator import decorator
from scipy.sparse.linalg.isolve.utils import make_system
from scipy._lib._util import _aligned_zeros
import numpy as np

_type_conv = {'f':'s', 'd':'d', 'F':'c', 'D':'z'}



def gmres(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, xtype=None, M=None, callback=None, restrt=None):
    """
    Use Generalized Minimal RESidual iteration to solve A x = b.
    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    Returns
    -------
    x : {array, matrix}
        The converged solution.
    info : int
        Provides convergence information:
          * 0  : successful exit
          * >0 : convergence to tolerance not achieved, number of iterations
          * <0 : illegal input or breakdown
    Other parameters
    ----------------
    x0 : {array, matrix}
        Starting guess for the solution (a vector of zeros by default).
    tol : float
        Tolerance to achieve. The algorithm terminates when either the relative
        or the absolute residual is below `tol`.
    restart : int, optional
        Number of iterations between restarts. Larger values increase
        iteration cost, but may be necessary for convergence.
        Default is 20.
    maxiter : int, optional
        Maximum number of iterations (restart cycles).  Iteration will stop
        after maxiter steps even if the specified tolerance has not been
        achieved.
    xtype : {'f','d','F','D'}
        This parameter is DEPRECATED --- avoid using it.
        The type of the result.  If None, then it will be determined from
        A.dtype.char and b.  If A does not have a typecode method then it
        will compute A.matvec(x0) to get a typecode.   To save the extra
        computation when A does not have a typecode attribute use xtype=0
        for the same type as b or use xtype='f','d','F',or 'D'.
        This parameter has been superseded by LinearOperator.
    M : {sparse matrix, dense matrix, LinearOperator}
        Inverse of the preconditioner of A.  M should approximate the
        inverse of A and be easy to solve for (see Notes).  Effective
        preconditioning dramatically improves the rate of convergence,
        which implies that fewer iterations are needed to reach a given
        error tolerance.  By default, no preconditioner is used.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(rk), where rk is the current residual vector.
    restrt : int, optional
        DEPRECATED - use `restart` instead.
    See Also
    --------
    LinearOperator
    Notes
    -----
    A preconditioner, P, is chosen such that P is close to A but easy to solve
    for. The preconditioner parameter required by this routine is
    ``M = P^-1``. The inverse should preferably not be calculated
    explicitly.  Rather, use the following template to produce M::
      # Construct a linear operator that computes P^-1 * x.
      import scipy.sparse.linalg as spla
      M_x = lambda x: spla.spsolve(P, x)
      M = spla.LinearOperator((n, n), M_x)
    ---

	   WORK(LDW,6+RESTRT)  WORK2(LDW2,RESTRT+2)
	 _ _ _ _ _ ____________   ______________
	| | | | | |            | |          | | |
	| | | | | |            | |          | | |
	| | | | | |            | |          | | |
	| | | | | |            | |          | | |
	|R|S|W|Y|A|  V(n,m+1)  | | H(m+1,m) |c|s|
	| | | | |V|            | |          | | |
	| | | | | |            | |          |---|
	| | | | | |            |  - - - - -  - -  LDW2 = m+1
	| | | | | |            |
	| | | | | |            |
	| | | | | |            |
         - - - - - - - - - - -  LDW = n
    ---
    """

    # Change 'restrt' keyword to 'restart'
    if restrt is None:
        restrt = restart
    elif restart is not None:
        raise ValueError("Cannot specify both restart and restrt keywords. "
                         "Preferably use 'restart' only.")

    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype)

    n = len(b)
    if maxiter is None:
        maxiter = n

    if restrt is None:
        restrt = n
    restrt = min(restrt, n)
    
    matvec = A.matvec
    psolve = M.matvec
    ltr = _type_conv[x.dtype.char]
    revcom = getattr(_iterative, ltr + 'gmresrevcom')
    stoptest = getattr(_iterative, ltr + 'stoptest2')

    resid = tol
    ndx1 = 1
    ndx2 = -1
    # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
    work = _aligned_zeros((6+restrt)*n,dtype=x.dtype)
    work2 = _aligned_zeros((restrt+1)*(2*restrt+2),dtype=x.dtype)
    ijob = 1
    info = 0
    ftflag = True
    bnrm2 = -1.0
    iter_ = maxiter
    old_ijob = ijob
    first_pass = True

    resid_ready = False
    iter_num = 1

    while True:
        olditer = iter_
        x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
           revcom(b, x, restrt, work, work2, iter_, resid, info, ndx1, ndx2, ijob)
        # if callback is not None and iter_ > olditer:
        #    callback(x)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)

        if (ijob == -1):  # gmres success, update last residual
            if resid_ready and callback is not None:
                callback(iter_num, resid, work, work2, ijob)
                resid_ready = False

            break
        elif (ijob == 1):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(x)
	    if callback is not None:
		callback(iter_num, resid, work, work2, ijob)
        elif (ijob == 2):
            work[slice1] = psolve(work[slice2])
            if not first_pass and old_ijob == 3:
		if callback is not None:
		    callback(iter_num, resid, work, work2, ijob)
                resid_ready = True
            first_pass = False
        elif (ijob == 3):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(work[slice1])
            if resid_ready and callback is not None:
                pass #callback(iter_num, resid, work, work2, ijob)
                resid_ready = False
            	iter_num = iter_num+1

        elif (ijob == 4):
            if ftflag:
                info = -1
                ftflag = False
            bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)

        old_ijob = ijob
        ijob = 2

        if iter_num > maxiter:
            break

    if info >= 0 and resid > tol:
        # info isn't set appropriately otherwise
        info = maxiter

   
    return postprocess(x)




