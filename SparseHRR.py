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

class SparseHRR:
    def __init__(self, size_or_arr, fillfactor=1):
        if isinstance(size_or_arr, int):
            size = size_or_arr
            num_filled = math.log(size, 2) * fillfactor
            probability = num_filled / size
            self.arr = numpy.random.binomial(1, probability, size)
            self.arr = numpy.fft.fft(self.arr)
            self.arr /= numpy.linalg.norm(self.arr)
        else:
            arr = size_or_arr
            self.arr = numpy.copy(arr)

        self.perm, self.invperm = _perm(len(self.arr))

    def normalize(self):
        return SparseHRR(self.arr / numpy.linalg.norm(self.arr))

    def permute_(self, perm):
        return SparseHRR(self.arr[perm])

    def permute(self):
        return self.permute_(self.perm)

    def invpermute(self):
        return self.permute_(self.invperm)

    def __mul__(self, other):
        """ Convolution/association """
        return SparseHRR(self.arr * other.arr)

    def __matmul__(self, other):
        """ Non-commutative/associative convolution/association """
        return self * other.permute()

    def __truediv__(self, other):
        """ Non-commutative/associative correlation/un-association """
        return self.decoderight(other)

    def decoderight(self, other):
        """ Non-commutative/associative correlation/un-association """
        return self * ~(other.invpermute())

    def decodeleft(self, other):
        """ Non-commutative/associative correlation/un-association """
        return (~other * self).invpermute()

    def __add__(self, other):
        """ Remember to normalize! """
        return SparseHRR(self.arr + other.arr)

    def __neg__(self):
        return SparseHRR(-self.arr)

    def __invert__(self):
        return SparseHRR(self.arr.conj())

    def __sub__(self, other):
        """ Remember to normalize! """
        return SparseHRR(self.arr - other.arr)

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
        cleanhrrs = [SparseHRR(memory.data[index]) for index in indices]
        cosinedistances = [hrr.similarity(cleanhrr) for cleanhrr in cleanhrrs]
        return cleanhrrs, cosinedistances

