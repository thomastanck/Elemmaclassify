import math
import numpy
import scipy.spatial
import functools

@functools.lru_cache()
def _perm(size):
    rng = numpy.random.default_rng(42)
    perm = rng.permutation(size)
    invperm = perm.argsort()
    return perm, invperm

class CircularHRR:
    def __init__(self, size_or_arr, seed=None):
        if isinstance(size_or_arr, int):
            size = size_or_arr
            if seed:
                rs = numpy.random.RandomState(seed)
            else:
                rs = numpy.random.default_rng()
            self.arr = numpy.exp(rs.uniform(math.tau, size=size) * 1j)
        else:
            arr = size_or_arr
            self.arr = numpy.copy(arr)

        self.perm, self.invperm = _perm(len(self.arr))

    def to_real_vec(self, dtype=None):
        realvec = numpy.empty(len(self.arr) * 2, dtype=dtype)
        realvec[0::2] = self.arr.real
        realvec[1::2] = self.arr.imag
        return realvec

    def normalize(self):
        return CircularHRR(self.arr / abs(self.arr))

    def permute_(self, perm):
        return CircularHRR(self.arr[perm])

    def permute(self):
        return self.permute_(self.perm)

    def invpermute(self):
        return self.permute_(self.invperm)

    def __mul__(self, other):
        """ Convolution/association or scalar multiplication """
        if isinstance(other, CircularHRR):
            return CircularHRR(self.arr * other.arr)
        elif numpy.isscalar(other):
            return CircularHRR(self.arr * other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """ Convolution/association or scalar multiplication """
        if isinstance(other, CircularHRR):
            return CircularHRR(self.arr * other.arr)
        elif numpy.isscalar(other):
            return CircularHRR(self.arr * other)
        else:
            return NotImplemented

    def __matmul__(self, other):
        """ Non-commutative/associative convolution/association """
        return self * other.permute()

    def __truediv__(self, other):
        """ Non-commutative/associative correlation/un-association """
        return self.decoderight(other)

    def decoderight(self, other):
        """ Non-commutative/associative correlation/un-association """
        return self * ~(other.permute())

    def decodeleft(self, other):
        """ Non-commutative/associative correlation/un-association """
        return (~other * self).invpermute()

    def __add__(self, other):
        """ Remember to normalize! """
        if other == 0:
            return CircularHRR(self.arr)
        return CircularHRR(self.arr + other.arr)

    def __radd__(self, other):
        """ Remember to normalize! """
        if other == 0:
            return CircularHRR(self.arr)
        return CircularHRR(self.arr + other.arr)

    def __neg__(self):
        return CircularHRR(-self.arr)

    def __invert__(self):
        return CircularHRR(self.arr.conj())

    def __sub__(self, other):
        """ Remember to normalize! """
        return CircularHRR(self.arr - other.arr)

    def similarity(self, other):
        return sum((self.arr / other.arr).real) / len(self.arr)

    def __repr__(self):
        return repr(self.arr)

    def __str__(self):
        return str(self.arr)

    @staticmethod
    def cleanupmemory(hrrs):
        return scipy.spatial.KDTree(numpy.copy([hrr.arr for hrr in hrrs]))

    @staticmethod
    def cleanup(memory, hrr, k=1):
        _, indices = memory.query(hrr.arr, k=k)
        if k == 1:
            indices = [indices]
        cleanhrrs = [CircularHRR(memory.data[index]) for index in indices]
        cosinedistances = [hrr.similarity(cleanhrr) for cleanhrr in cleanhrrs]
        return cleanhrrs, cosinedistances
