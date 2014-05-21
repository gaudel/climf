"""
CLiMF Collaborative Less-is-More Filtering, a variant of latent factor CF
which optimises a lower bound of the smoothed reciprocal rank of "relevant"
items in ranked recommendation lists.  The intention is to promote diversity
as well as accuracy in the recommendations.  The method assumes binary
relevance data, as for example in friendship or follow relationships.

CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering
Yue Shi, Martha Larson, Alexandros Karatzoglou, Nuria Oliver, Linas Baltrunas, Alan Hanjalic
ACM RecSys 2012
"""

from math import exp, log
import numpy as np
import random
from climf_fast import climf_fast, safe_climf_fast, CSRDataset, compute_mrr_fast

def _make_dataset(X):
    """Create ``Dataset`` abstraction for sparse and dense inputs."""
    y_i = np.ones(X.shape[0], dtype=np.float64, order='C')
    sample_weight = np.ones(X.shape[0], dtype=np.float64, order='C') # ignore sample weight for the moment
    dataset = CSRDataset(X.data, X.indptr, X.indices, y_i, sample_weight)
    return dataset

class CLiMF:
    def __init__(self, dim=10, lbda=0.001, gamma=0.0001, max_iters=5, verbose=True,
                 shuffle=True, seed=28):
        self.dim = dim
        self.lbda = lbda
        self.gamma = gamma
        self.max_iters = max_iters
        self.verbose = verbose
        self.shuffle = 1 if shuffle else 0
        self.seed = seed

    def fit(self, X, safe=False):
        data = _make_dataset(X)
        self.U = 0.01*np.random.random_sample(size=(X.shape[0], self.dim))
        self.V = 0.01*np.random.random_sample(size=(X.shape[1], self.dim))

        num_train_sample_users = min(X.shape[0],100)
        train_sample_users = np.array(random.sample(xrange(X.shape[0]),num_train_sample_users), dtype=np.int32)
        sample_user_data = np.array([np.array(X.getrow(i).indices, dtype=np.int32) for i in train_sample_users])
        
        if not safe :
            climf_fast(data, self.U, self.V, self.lbda, self.gamma, self.dim, 
                   self.max_iters, self.shuffle, self.seed, train_sample_users, sample_user_data)
        else :
            safe_climf_fast(data, self.U, self.V, self.lbda, self.gamma, self.dim, 
                   self.max_iters, self.shuffle, self.seed, train_sample_users, sample_user_data)

    def compute_mrr(self, testdata):
        return compute_mrr_fast(np.array(range(testdata.shape[0]), dtype=np.int32), np.array([np.array(testdata.getrow(i).indices, dtype=np.int32) for i in range(testdata.shape[0])]), self.U, self.V)
        
if __name__=='__main__':

    from optparse import OptionParser
    import random
    from scipy.io.mmio import mmread

    parser = OptionParser()
    parser.add_option('--train',dest='train',help='training dataset (matrixmarket format)')
    parser.add_option('--test',dest='test',help='optional test dataset (matrixmarket format)')
    parser.add_option('-d','--dim',dest='D',type='int',default=10,help='dimensionality of factors (default: %default)')
    parser.add_option('-l','--lambda',dest='lbda',type='float',default=0.001,help='regularization constant lambda (default: %default)')
    parser.add_option('-g','--gamma',dest='gamma',type='float',default=0.0001,help='gradient ascent learning rate gamma (default: %default)')
    parser.add_option('--max_iters',dest='max_iters',type='int',default=25,help='max iterations (default: %default)')
    parser.add_option("--safe", action="store_true", dest="safe",help='use safe version of climf (default: %default)')
    
    (opts,args) = parser.parse_args()
    if not opts.train or not opts.D or not opts.lbda or not opts.gamma:
        parser.print_help()
        raise SystemExit

    data = mmread(opts.train).tocsr()  # this converts a 1-indexed file to a 0-indexed sparse array
    if opts.test:
        testdata = mmread(opts.test).tocsr()


    cf = CLiMF(lbda=opts.lbda, gamma=opts.gamma, dim=opts.D, max_iters=opts.max_iters)
    cf.fit(data, safe=opts.safe)
#    if opts.test:
#        num_test_sample_users = min(testdata.shape[0],1000)
#        test_sample_users = random.sample(xrange(testdata.shape[0]),num_test_sample_users)
#        print 'test mrr  = {0:.4f}'.format(compute_mrr(testdata,U,V,test_sample_users))
#    in for-loop
#        if opts.test:
#            print 'test mrr  = {0:.4f}'.format(compute_mrr(testdata,U,V,test_sample_users))

    if opts.test:
        print "Test MRR: %.8f" % cf.compute_mrr(testdata)
    print 'U',cf.U
    print 'V',cf.V
