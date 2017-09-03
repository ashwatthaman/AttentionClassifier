from chainer import cuda
import numpy
from chainer import Variable
import numpy
from chainer import cuda
from chainer import link
import numpy as np

class XP:
    __lib = None

    @staticmethod
    def set_library(args):
        if args.use_gpu:
            XP.__lib = cuda.cupy
            cuda.get_device(args.gpu_device).use()
        else:
            XP.__lib = numpy

    @staticmethod
    def __zeros(shape, dtype):
        return Variable(XP.__lib.zeros(shape, dtype=dtype))

    @staticmethod
    def fzeros(shape):
        return XP.__zeros(shape, XP.__lib.float32)

    @staticmethod
    def __array(array, dtype):
        return Variable(XP.__lib.array(array, dtype=dtype))

    @staticmethod
    def iarray(array):
        return XP.__array(array, XP.__lib.int32)

    @staticmethod
    def farray(array):
        return XP.__array(array, XP.__lib.float32)

def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print('Copy %s' % child.name)

def predictRandom(prob):
    probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
    # print("probab:{}".format(probability))
    probability /= np.sum(probability)
    index = np.random.choice(range(len(probability)), p=probability)
    return index
