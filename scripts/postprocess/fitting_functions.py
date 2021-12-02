""" Script where we define a custom curve_fit function by modifying the
scipy.optimize.curve_fit function (as of Scipy 1.4.1).
Added features:
    - Passing extra arguments (via the func_kwargs dict) to the fitted function
    - Support fitting vector fields of one scalar parameter (time); no need to
        give duplicated time points and flatten the output of the vector function
    - L1 regularization term (\lambda \sum_i |\theta_i|) automatically added
        to the cost function; just specify a regularization rate \lambda,
        no need to add fictitious points in the fitted function.
    - Regularize each parameter around a specified non-zero value
    with the offsets argument.
    - Returning the jacobian of the cost function as well as the covariance.
"""
import numpy as np
import scipy as sp

# Import scipy functions used by curve_fit
from scipy.optimize.minpack import _initialize_feasible, _wrap_func, _wrap_jac, prepare_bounds
from scipy.optimize import least_squares, OptimizeWarning
from scipy.linalg import cholesky, LinAlgError, svd

### Custom wrappers to deal with scalar-to-vector functions and regularization
# This is exactly the scipy wrapper but with func_kwargs.
def _wrap_func_simple(func, xdata, ydata, transform, func_kwargs={}):
    # Slight modification of scipy's wrapper to pass kwargs to func
    # and to deal with m dimensional functions
    if transform is None:
        def func_wrapped(params):
            return (func(xdata, *params, **func_kwargs) - ydata).flatten()
    elif transform.ndim == 1:
        def func_wrapped(params):
            return transform * (func(xdata, *params, **func_kwargs) - ydata).flatten()
    else:
        def func_wrapped(params):
            residuals = (func(xdata, *params, **func_kwargs) - ydata).flatten()
            return solve_triangular(transform, residuals, lower=True)
    return func_wrapped

def _wrap_func_regul(func, xdata, ydata, transform, reg_rate, offsets, func_kwargs={}):
    # Func can be a m-dimensional vector function, return a m x len(xdata)
    # Then ydata must also have that shape.
    if transform is None:
        # Concatenate the residuals with reg_rate*sqrt(abs((parameters - offsets)))
        def func_wrapped(params):
            residuals = func(xdata, *params, **func_kwargs) - ydata
            # Assuming the list of residuals has the same length as the list of parameters
            # Should be checked in curve_fit_jac
            regul = reg_rate * np.sqrt(np.abs(np.asarray(params) - offsets))
            return np.concatenate([residuals.flatten(), regul])

    elif transform.ndim == 1:
        # If we have a m-dimensional function and independent variance for each,
        # make sure transform is a flattened array with all points of one
        # dimension next to each other
        def func_wrapped(params):
            residuals = transform * (func(xdata, *params, **func_kwargs) - ydata).flatten()
            regul = reg_rate * np.sqrt(np.abs(np.asarray(params) - offsets))
            return np.concatenate([residuals, regul])
    else:
        # Chisq = (y - yd)^T C^{-1} (y-yd)
        # transform = L such that C = L L^T
        # C^{-1} = L^{-T} L^{-1}
        # Chisq = (y - yd)^T L^{-T} L^{-1} (y-yd)
        # Define (y-yd)' = L^{-1} (y-yd) by solving
        # L (y-yd)' = (y-yd)
        # and minimize (y-yd)'^T (y-yd)'
        # Make sure transform is a (m * len(xdata)) x (m * len(xdata)) matrix
        # if func is m-dimensional. So it's a block matrix where each block
        # is the covariance between points on one of the dimensions and another
        # C = [C_{11} C_{12} ...  C_{1m}]
        #     [C_{21} C_{22} ...  C_{2m}]
        #     [...    ...    ...   ...  ]
        #     [C_{m1} C_{m2} ...  C_{mm}]
        # where C_{ij} is a len(xdata) x len(xdata) matrix
        def func_wrapped(params):
            residuals = (func(xdata, *params, **func_kwargs) - ydata).flatten()
            residuals = solve_triangular(transform, residuals, lower=True)
            regul = reg_rate * np.sqrt(np.abs(np.asarray(params) - offsets))
            return np.concatenate([residuals, regul])

    # Return the appropriate wrapper
    return func_wrapped

# This is exactly the scipy default function but with func_kwargs.
def _wrap_jac_simple(jac, xdata, transform, func_kwargs={}):
    if transform is None:
        def jac_wrapped(params):
            return jac(xdata, *params, **func_kwargs)
    elif transform.ndim == 1:
        def jac_wrapped(params):
            return transform[:, np.newaxis] * np.asarray(jac(xdata, *params, **func_kwargs))
    else:
        def jac_wrapped(params):
            return solve_triangular(transform, np.asarray(jac(xdata, *params, **func_kwargs)), lower=True)
    return jac_wrapped

### Custom curve_fit returns the jacobian too and uses the upgraded wrappers.
def curve_fit_jac(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
              check_finite=True, bounds=(-np.inf, np.inf), method=None,
              jac=None, reg_rate=None, offsets=None, func_kwargs={}, **kwargs):
    """
    Use non-linear least squares to fit a function, f, to data.

    Assumes ``ydata = f(xdata, *params) + eps``

    Parameters
    ----------
    f : callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
        It can be a vector function, returning a m x len(xdata) matrix
    xdata : array_like or object
        The independent variable where the data is measured.
        Should usually be an M-length sequence or an (k,M)-shaped array for
        functions with k predictors, but can actually be any object.
    ydata : array_like
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
    p0 : array_like, optional
        Initial guess for the parameters (length N).  If None, then the
        initial values will all be 1 (if the number of parameters for the
        function can be determined using introspection, otherwise a
        ValueError is raised).
    sigma : None or M-length sequence or MxM array, optional
        Determines the uncertainty in `ydata`. If we define residuals as
        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
        depends on its number of dimensions:

            - A 1-d `sigma` should contain values of standard deviations of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = sum((r / sigma) ** 2)``.

            - A 2-d `sigma` should contain the covariance matrix of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = r.T @ inv(sigma) @ r``.

              .. versionadded:: 0.19

        None (default) is equivalent of 1-d `sigma` filled with ones.
    absolute_sigma : bool, optional
        If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.

        If False, only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    check_finite : bool, optional
        If True, check that the input arrays do not contain nans of infs,
        and raise a ValueError if they do. Setting this parameter to
        False may silently produce nonsensical results if the input arrays
        do contain nans. Default is True.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to no bounds.
        Each element of the tuple must be either an array with the length equal
        to the number of parameters, or a scalar (in which case the bound is
        taken to be the same for all parameters.) Use ``np.inf`` with an
        appropriate sign to disable bounds on all or some parameters.

        .. versionadded:: 0.17
    method : {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimization.  See `least_squares` for more details.
        Default is 'lm' for unconstrained problems and 'trf' if `bounds` are
        provided. The method 'lm' won't work when the number of observations
        is less than the number of variables, use 'trf' or 'dogbox' in this
        case.

        .. versionadded:: 0.17
    jac : callable, string or None, optional
        Function with signature ``jac(x, ...)`` which computes the Jacobian
        matrix of the model function with respect to parameters as a dense
        array_like structure. It will be scaled according to provided `sigma`.
        If None (default), the Jacobian will be estimated numerically.
        String keywords for 'trf' and 'dogbox' methods can be used to select
        a finite difference scheme, see `least_squares`.
        If f is a m dimensional function and jac is callable,
        make sure the rows of jac are ordered with all points of
        one dimension next to each other.

        .. versionadded:: custom
    reg_rate : float or None, optional
        Magnitude of the regularization term, which is the L1 norm of parameters
        centered on offsets.

        .. versionadded:: custom
    offsets : list, np.ndarray or None, optional
        Values of the parameters that minimize the regularization term,
        reg_rate * |parameters - offsets| .

        .. versionadded:: custom
    func_kwargs : dict, optional
        Extra keyword arguments to pass to func after the parameters, default {}.

        .. versionadded:: 0.18
    kwargs
        Keyword arguments passed to `leastsq` for ``method='lm'`` or
        `least_squares` otherwise.

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals of ``f(xdata, *popt) - ydata`` is minimized

    pcov : 2d array
        The estimated covariance of popt. The diagonals provide the variance
        of the parameter estimate. To compute one standard deviation errors
        on the parameters use ``perr = np.sqrt(np.diag(pcov))``.

        How the `sigma` parameter affects the estimated covariance
        depends on `absolute_sigma` argument, as described above.

        If the Jacobian matrix at the solution doesn't have a full rank, then
        'lm' method returns a matrix filled with ``np.inf``, on the other hand
        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
        the covariance matrix.

    jac : 2d array
        The estimated jacobian matrix at the optimal solution popt.
        The hessian can be computed as J^T J, where J is the jacobian.
        We return the jacobian so rows corresponding to regularization
        residuals can be removed before computing the hessian, which
        then only reflects the influence of parameters on the data fit.
        The (pseudo-)inverse of the hessian is the covariance matrix,
        modulo a multiplicative constant, the sample standard deviation.

    Raises
    ------
    ValueError
        if either `ydata` or `xdata` contain NaNs, or if incompatible options
        are used.

    RuntimeError
        if the least-squares minimization fails.

    OptimizeWarning
        if covariance of the parameters can not be estimated.

    See Also
    --------
    least_squares : Minimize the sum of squares of nonlinear functions.
    scipy.stats.linregress : Calculate a linear least squares regression for
                             two sets of measurements.

    Notes
    -----
    With ``method='lm'``, the algorithm uses the Levenberg-Marquardt algorithm
    through `leastsq`. Note that this algorithm can only deal with
    unconstrained problems.

    Box constraints can be handled by methods 'trf' and 'dogbox'. Refer to
    the docstring of `least_squares` for more information.
    """
    # Functions needed: _getargspec, _initialize_feasible, prepare_bounds, cholesky, _wrap_func, _wrap_jac,
    # svd, least_squares, leastsq, warnings
    if p0 is None:
        # determine number of parameters by inspecting the function
        from scipy._lib._util import getargspec_no_self as _getargspec
        args, varargs, varkw, defaults = _getargspec(f)
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n = len(args) - 1
    else:
        p0 = np.atleast_1d(p0)
        n = p0.size

    lb, ub = prepare_bounds(bounds, n)
    if p0 is None:
        p0 = _initialize_feasible(lb, ub)

    bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
    if method is None:
        if bounded_problem:
            method = 'trf'
        else:
            method = 'lm'

    if method == 'lm' and bounded_problem:
        raise ValueError("Method 'lm' only works for unconstrained problems. "
                         "Use 'trf' or 'dogbox' instead.")

    # optimization may produce garbage for float32 inputs, cast them to float64

    # NaNs can not be handled
    if check_finite:
        ydata = np.asarray_chkfinite(ydata, float)
    else:
        ydata = np.asarray(ydata, float)

    if isinstance(xdata, (list, tuple, np.ndarray)):
        # `xdata` is passed straight to the user-defined `f`, so allow
        # non-array_like `xdata`.
        if check_finite:
            xdata = np.asarray_chkfinite(xdata, float)
        else:
            xdata = np.asarray(xdata, float)

    if ydata.size == 0:
        raise ValueError("`ydata` must not be empty!")

    # Determine type of sigma
    if sigma is not None:
        sigma = np.asarray(sigma)

        # if 1-d, sigma are errors, define transform = 1/sigma
        if sigma.shape == (ydata.size, ):
            transform = 1.0 / sigma
        # if 2-d, sigma is the covariance matrix,
        # define transform = L such that L L^T = C
        elif sigma.shape == (ydata.size, ydata.size):
            try:
                # scipy.linalg.cholesky requires lower=True to return L L^T = A
                transform = cholesky(sigma, lower=True)
            except LinAlgError:
                raise ValueError("`sigma` must be positive definite.")
        else:
            raise ValueError("`sigma` has incorrect shape.")
    else:
        transform = None

    # Function wrapper that returns residuals and regularization term
    if reg_rate is None or abs(reg_rate) < 1e-16:  # if reg_rate is negative, will be squared, OK
        if offsets is not None:
            print("No regularization applied, reg_rate={} invalid".format(reg_rate))
        # Simple wrapper without regularization, no kwargs
        func = _wrap_func_simple(f, xdata, ydata, transform, func_kwargs)
    else:
        if offsets is None:
            print("offsets is None, so no regularization applied")
            func = _wrap_func_simple(f, xdata, ydata, transform, func_kwargs)
        else:
            func = _wrap_func_regul(f, xdata, ydata, transform, reg_rate, offsets, func_kwargs)

    # TODO: different wrapper for the jacobian too if there is regulation
    if callable(jac):
        jac = _wrap_jac(jac, xdata, transform, func_kwargs)
    elif jac is None:
        jac = '2-point'

    if 'args' in kwargs:
        # The specification for the model function `f` does not support
        # additional arguments. Refer to the `curve_fit` docstring for
        # acceptable call signatures of `f`.
        raise ValueError("'args' is not a supported keyword argument.")

    ## From this point on, we modify the package's curve_fit, to use only least_squares.
    # Rename maxfev (leastsq) to max_nfev (least_squares), if specified.
    if 'max_nfev' not in kwargs:
        kwargs['max_nfev'] = kwargs.pop('maxfev', None)

    res = least_squares(func, p0, jac=jac, bounds=bounds, method=method,
                        **kwargs)

    if not res.success:
        raise RuntimeError("Optimal parameters not found: " + res.message)

    ysize = len(res.fun)
    cost = 2 * res.cost  # res.cost is half sum of squares!
    popt = res.x

    # Do not invert the jacobian, compute the hessian at first order
    # J^T J, which is exactly the Fisher information matrix
    #hess = np.dot(res.jac.T, res.jac)
    # Instead return the jacobian, so rows can be removed from it
    # before computing the hessian.

    # Compute the covariance matrix as well.
    if np.any(np.isnan(res.jac)):
        pcov = None
    else:
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        return_full = False
    warn_cov = False
    if pcov is None:
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.nan)
        warn_cov = True
    elif not absolute_sigma:
        if ysize > p0.size:
            s_sq = cost / (ysize - p0.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(np.nan)
            warn_cov = True

    if warn_cov:
        warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning)

    return popt, pcov, res.jac
