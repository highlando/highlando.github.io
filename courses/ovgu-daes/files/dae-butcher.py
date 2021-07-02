import numpy as np

fac = np.math.factorial
np.set_printoptions(precision=3, suppress=True)


def get_kappa(jay, betat=None, gamma=None, ainv=None, verbose=False):
    n = ainv.shape[0]
    ones = np.ones((n, 1))
    ainvk = np.eye(n)  # ainv^0
    ainvjay = np.eye(n)  # ainv^0
    kcheck = True
    for iy in np.arange(jay):
        ainvjay = ainvjay.dot(ainv)
    for kay in np.arange(1, jay):
        ainvk = ainv.dot(ainvk)
        leftside = (betat.dot(ainvk)).dot(ones)
        rightside = (betat.dot(ainvjay)).dot(gamma**kay)/fac(jay-kay)
        kcheck = np.allclose(leftside, rightside)
        if verbose:
            print('jay={2}/kay={0}: check={1}'.format(kay, kcheck, jay))
        if not kcheck:
            return None
    kappajay = jay-1
    kay = jay
    while kcheck:
        leftside = (betat.dot(ainvjay)).dot(gamma**kay)
        rightside = fac(kay)/fac(kay-jay+1)
        kcheck = np.allclose(leftside, rightside)
        if verbose:
            print('jay={2}/kay={0}: check={1}'.format(kay, kcheck, jay))
        if not kcheck:
            return kappajay
        if kappajay == 99:
            return kappajay
        kappajay = kay
        kay += 1


if __name__ == '__main__':

    nu = 3

    name = 'Gauss-2'
    z = np.sqrt(3)/6
    beta = np.array([.5, .5]).reshape((2, 1))
    gamma = np.array([.5-z, .5+z]).reshape((2, 1))
    biga = np.array([.25, .25-z, .25+z, .25]).reshape((2, 2))
    ainv = 12*np.array([.25, -.25+z, -.25-z, .25]).reshape((2, 2))

    name = 'RadauIIa-1'
    beta = np.array([1.]).reshape((1, 1))
    gamma = np.array([1.]).reshape((1, 1))
    ainv = np.array([1.]).reshape((1, 1))
    biga = np.array([1.]).reshape((1, 1))

    name = 'RadauIIa-2'
    dzi = 1./12
    beta = np.array([9*dzi, 3*dzi]).reshape((2, 1))
    gamma = np.array([4*dzi, 1.]).reshape((2, 1))
    biga = np.array([5*dzi, -dzi, 9*dzi, 3*dzi]).reshape((2, 2))
    ainv = 6*np.array([3*dzi, dzi, -9*dzi, 5*dzi]).reshape((2, 2))

    print(name)
    print('\nbeta.T:\n', beta.T)
    print('\ngamma:\n', gamma)
    print('\nbiga:\n', biga)
    print('\nainv:\n', ainv)
    print('\nbiga*ainv:\n', biga.dot(ainv), '\n')

    for jay in np.arange(nu):
        kapj = get_kappa(jay+1, betat=beta.T, gamma=gamma, ainv=ainv,
                         verbose=False)
        print('kappa_{0}: {1}'.format(jay+1, kapj))
