import scipy.sparse as sp
from scipy.sparse.linalg import splu
import numpy as np

class GRP(object):
	def __init__(self, alpha, beta, t, d):
		self.alpha = alpha
		self.beta = beta
		self.t = t
		self.d = d

		self.N = len(t)
		self.m = len(alpha)

	def assemble_matrix(self):
		self.gamma = np.empty((self.m,self.N-1))
		for i in range(self.m):
			for j in range(self.N-1):
				self.gamma[i,j]	=	np.exp(-self.beta[i]*abs(self.t[j]-self.t[j+1]))

		twom = 2*self.m
		self.blocknnz = 6*self.m+1
		self.nBlockSize = twom+1
		self.M = self.N*self.nBlockSize-twom
		self.nnz = (self.N-1)*self.blocknnz+(self.N-2)*twom+1

		self.nBlockStart = []
		for k in range(self.N):
			self.nBlockStart.append(k*self.nBlockSize)

		self.row, self.col, self.data = [], [], []
		for nBlock in range(self.N-1):
		#	The starting row and column for the blocks.
		#	Assemble the diagonal first.
			self.row.append(self.nBlockStart[nBlock]) 
			self.col.append(self.nBlockStart[nBlock])
			self.data.append(self.d[nBlock])
			for k in range(self.m):
				self.row.append(self.nBlockStart[nBlock]+k+1) 
				self.col.append(self.nBlockStart[nBlock])
				self.data.append(self.gamma[k,nBlock])
				# triplets.push_back(Eigen::Triplet<double>(nBlockStart[nBlock],nBlockStart[nBlock]+k+1,gamma(k,nBlock)));
				self.row.append(self.nBlockStart[nBlock]) 
				self.col.append(self.nBlockStart[nBlock]+k+1)
				self.data.append(self.gamma[k,nBlock])
				# triplets.push_back(Eigen::Triplet<double>(nBlockStart[nBlock]+m+k+1,nBlockStart[nBlock]+twom+1,alpha(k)));
				self.row.append(self.nBlockStart[nBlock]+self.m+k+1) 
				self.col.append(self.nBlockStart[nBlock]+twom+1)
				self.data.append(self.alpha[k])
				# triplets.push_back(Eigen::Triplet<double>(nBlockStart[nBlock]+twom+1,nBlockStart[nBlock]+m+k+1,alpha(k)));
				self.row.append(self.nBlockStart[nBlock]+twom+1) 
				self.col.append(self.nBlockStart[nBlock]+self.m+k+1)
				self.data.append(self.alpha[k])
				# triplets.push_back(Eigen::Triplet<double>(nBlockStart[nBlock]+k+1,nBlockStart[nBlock]+k+m+1,-1.0));
				self.row.append(self.nBlockStart[nBlock]+k+1) 
				self.col.append(self.nBlockStart[nBlock]+k+self.m+1)
				self.data.append(-1.0)
				# triplets.push_back(Eigen::Triplet<double>(nBlockStart[nBlock]+k+m+1,nBlockStart[nBlock]+k+1,-1.0));
				self.row.append(self.nBlockStart[nBlock]+k+self.m+1) 
				self.col.append(self.nBlockStart[nBlock]+k+1)
				self.data.append(-1.0)

		# triplets.push_back(Eigen::Triplet<double>(M-1,M-1,d(N-1)));
		self.row.append(self.M-1)
		self.col.append(self.M-1)
		self.data.append(self.d[self.N-1])

		# Assebmles the supersuperdiagonal identity blocks.
		for nBlock in range(self.N-2):
			for k in range(self.m):
				# triplets.push_back(Eigen::Triplet<double>(nBlockStart[nBlock]+k+m+1,nBlockStart[nBlock]+twom+k+2,gamma(k,nBlock+1)));
				self.row.append(self.nBlockStart[nBlock]+self.m+k+1) 
				self.col.append(self.nBlockStart[nBlock]+twom+k+2)
				self.data.append(self.gamma[k,nBlock+1])
				# triplets.push_back(Eigen::Triplet<double>(nBlockStart[nBlock]+twom+k+2,nBlockStart[nBlock]+k+m+1,gamma(k,nBlock+1)));
				self.row.append(self.nBlockStart[nBlock]+twom+k+2) 
				self.col.append(self.nBlockStart[nBlock]+self.m+k+1)
				self.data.append(self.gamma[k,nBlock+1])

		self.Aex = sp.csc_matrix((np.array(self.data), (np.array(self.row), np.array(self.col))), shape=(self.M,self.M))

	def factor(self):
		self.factorize = splu(self.Aex)

	def solve(self, rhs):
		rhsex = np.zeros(self.M)
		rhsex[::self.nBlockSize] = rhs
		solex = self.factorize.solve(rhsex)

		return solex[::self.nBlockSize]

	def logdeterminant(self):
		return np.log(abs(self.factorize.U.diagonal())).sum()
