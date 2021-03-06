
## This seems to give a much more 'OpenCL'-like error message now... : 
# THEANO_FLAGS=mode=FAST_RUN,device=opencl0:0,floatX=float32 optirun python misc/test_opencl.py 

## This works with the opencl-stdint branch :
# THEANO_FLAGS=mode=FAST_RUN,device=opencl0:0,floatX=float32,exception_verbosity=high,optimizer=fast_compile  optirun python misc/test_opencl.py 

## Standard CPU version :
# python misc/test_opencl.py 


from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print f.maker.fgraph.toposort()
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print 'Looping %d times took' % iters, t1 - t0, 'seconds'
print 'Result is', r
if numpy.any([isinstance(x.op, T.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print 'Used the cpu'
else:
    print 'Used the gpu'
