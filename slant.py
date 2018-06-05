import PriorityQueue from PriorityQueue
import math
import numpy.random as rnd
import numpy as np
from numpy import linalg as LA
from scipy import linalg as scipyLA
from a import graph
import heapq
from collections import deque
class slant:
	def __init__(self,obj):
		self.num_node = obj.num_node
		self.num_sentiment_val = obj.num_sentiment_val # *****************
		self.nodes = obj.nodes
		self.graph=obj.graph
		self.train= obj.train
		self.test = obj.test

		# alpha
		# A
		# mu
		# B

	# def spectral_proj_grad(self,h):
	# 	k=0
	# 	x=[mu_0 B_0] # check
	# 	f_array=deque([],maxlen=h)
	# 	while True:
	# 		alpha_bar=min([alpha_max,alpha_bb]) #
	# 		d=self.proj(x-alpha_bar*self.grad_f(x))-x # 
	# 		f_array.appendleft(self.f(x)) #
	# 		f_b=max(f_array) # save only h elm
	# 		alpha=1
	# 		while lhs < f_b + v*alpha*self.grad_f(x).dot(d): # lhs
	# 			#modify alpha
	# 		x=x+alpha*d
	# 		s=alpha*d
	# 		y=alpha*B*d#
	# 		alpha_bb = 0 ## write 
	# 		k=k+1
	# 		if  LA.norm(d)<epsilon:
	# 			break



	def find_mu_B(self):
		self.mu=np.array(self.nuser)
		self.B={}
		for u in range(self.nuser):
			self.mu[u],self.B[u] =self.spectral_proj_grad()
	def solve_least_square(self,A,b):
		At = A.transpose()
		x=LA.solve(np.matmul(At,A),np.dot(At,b))
		return x[0],x[1:]
	def find_alpha_A(self):
		# maintain msg counter for each user
		# map users to 0 to nusers
		ind_4_V=np.zero(self.nuser)
		tau={}
		for u in range(nuser):
			tau[u]=[]
		for t,u,m in H: #
			tau[u].append([t,u,m])
			ind_4_v[u]=ind_4_v[u]+1
		for u in range(nuser):
			i=0
			S=tau[u]
			for nbr in graph[u]:
				S=self.merge(S,tau[nbr])#
				x_last=0
				t_last=0
				m_no=0
				for t,v,m in S:
					if v==u:
						x=x_last*math.exp(-w*(t-t_last))
						g(m_no,v)=x
						y(m_no)=m
						m_no=m_no+1
					else:
						x=x_last*math.exp(-w*(t-t_last))
					t_last=t
					x_last=x
			alpha[u],A[u]=self.solve_least_square(g,y,lda)
		# nmsg
		# M=np.array(nuser,nmsg.max(),3)
		# for u in self.graph.vertices:
		# 	M[u,:,:1]=np.asarray([train[i,1:] if train[i,0]==u for i in self.ntrain])
		# 	M[u,:,2]=np.multiply(M[u,:,0],np.exp(w*M[u,:,1]))
		# alpha=np.array(nuser)
		# A=[]
		# for u in self.graph.vertices:
		# 	M_hat=np.array(nmsg,len(self.edges[u])+1)
		# 	arg1=M_hat.transpose().dot(M_hat)
		# 	arg2=M_hat.transpose().dot(M[u,:,0])
		# 	res = LA.solve(arg1,arg2)
		# 	alpha[u]=res[0]
		# 	A.append(list(res[1:]))
		# 	alpha[u],A[u]=self.solve_least_square(A,b)
		return alpha,A
	def estimate_param(self,train):
		self.mu =  rnd.uniform(size = self.nuser)
		self.A = self.B = []
		for user in self.graph.nodes:
			num_nbr = len(self.graph.edges[user])
			A_user = rnd.uniform(size = num_nbr ) # chec whether return ndarray
			B_user = rnd.uniform( size = num_nbr )
			self.A.append(A_user)
			self.B.append(B_user)
		self.alpha = rnd.uniform( low = 0 , high = self.num_sentiment_val, size = self.nuser )
		
		# self.mu,self.B = self.find_mu_B()
		# self.alpha, self.A = self.find_alpha_A()
	def predict(self,test):
		# test is a dictionary
		# test['user']=[set of msg posted by that user]
		# each user is a key 
		# each user has a list of messages attached to it.
		# for each message ,
		# sample a set of msg first
		# predict the msg of that user at that time
		# save the prediction in a dictionary called prediction 
		# return a set of predicted msg
		self.performance.MSE = self.get_MSE(predict_test, test[:,1]) # define performance
		self.performance.FR = self.get_FR(predict_test, test[:,1])  
	def simulation_based_forecast(self,T):
		H=forecast_sampling_events(T)
	def forecast_sampling_events(self,T):#
		# test message set , parameters learnt from train , T  , graph ( number of node and adj list )
		# sample events 
		# sample events for each user
		# until we reach T , we generate min t , generate corresponding event ,  update all neighbours 
		# predict message sentiment for each msg in test set
		# return prediction 
		opn_update=[np.zero(self.nuser) self.alpha] 
		int_update = [np.zero(self.nuser) self.mu] 
		H=[]
		tQ=np.array(self.nuser)
		for u in range(nuser):
			tQ[u]=self.sample_event(self.mu[u],0,u) 
		Q=PriorityQueue(tQ)
		while t<T:#** define t and T 
			t_new,u=Q.extract_prior()# do not we need o put back t_new,u 
			# t_old=opn_update_time[u]
			# x_old=opn_update_val[u]
			[t_old,x_old] = opn_update[u][:]
			x_new=self.alpha[u]+(x_old-self.alpha[u])*np.exp(-self.w*(t_new-t_old))
			# opn_update_time[u]=t_new
			# opn_update_val[u]=x_new
			opn_update[u]=[t_new,x_new]
			m=sample_gaussian(x_new)#**
			H.append([t_new,m,u])
			# update neighbours
			for nbr in self.graph[u]:
				# change above 
				[t_old,lda_old] = int_update[nbr][:]
				lda_new = self.mu[nbr]+(lda-self.mu[nbr])*np.exp(-self.w*(t_new-t_old))+self.B[u][nbr]# use sparse matrix
				int_update[nbr]=np.array([t_new,lda_new])#**
				t_old,x_old=list(opn_update[nbr])
				x_new=self.alpha[nbr]+(x_old-self.alpha[nbr])*np.exp(-self.w*(t_new-t_old))+self.A[u][nbr]*m
				opn_update[nbr]=np.array([t_new,x_new])
				t_nbr=self.sample_event(lda_new,t_new,nbr)
				self.Q.update_key(t_nbr,nbr) 
			t=t_new
		return H
	def sample_event(self,lda,t,v):
		lda_bar=lda
		t_new=t
		while t_new<self.T: #* get T 
			u=sample_uniform(0,1)#* 
			t_new=t_new-math.log(u)/lda_bar
			lda_new=self.mu[v]+(lda-self.mu[v])*np.exp(-self.w*(t_new-t))
			d=sample_uniform(0,1)# **
			if d*lda_bar<lda:
				return t_new
			else
				lda_bar=lda_new
		return t_new
	def get_FR(self,s,t):
		return (len(s)-np.count_nonzero(np.sign(s)+np.sign(t)))/len(s)
	def get_MSE(self,s,t):
		return ((s-t)**2).mean()
	def analytical_opinion_forecast_poisson(self,train, test): # analytical using poisson
		# create nbr[u]
		time = train[:,2] - t0 
		temp = np.exp(-time*w).dot(train[:,1])
		sentiment =np.array(nuser)
		for u in range(nuser):
			sentiment[u]=sum(np.asarray([temp[i] if train[i,0]==u for i in range(ntrain)]))
		
		x_t0=self.alpha[u]+A[u,self.graph.edges[u]].dot(sentiment[self.graph.edges[u]])
		mat_arg=self.A.dot(np.diag(self.mu))-w*np.eye(nuser)
		inv_mat = self.inverse_mat(mat_arg)
		res=np.zeros(ntest)
		for m in range(ntest):
			exp_mat = scipyLA.expm(self.delta_t*mat_arg)
			sub_term2 = (exp_mat-eye(nuser))*self.alpha
			res[m] = exp_mat[u]*x_t0+w[u]*inv_mat[u]*sub_term2
		return res		
def main():
	input_file = 'input_data'
	# load data 
	with open(input_file+'.pkl','rb') as f:
		data = pickle.load(f)
		# data is object with three component
		# graph , 
		# train ( dictionary of msg )
		# test ( dictionary of msg )
	# slant_obj = slant(data.graph)
	# slant_obj.estimate_param(data.train)
	# result = slant_obj.predict(data.test)
	#******************** test *******************************************
	print type(rnd.uniform(size = num_nbr ))
	# def inverse_mat(self,mat):
	# 	return LA.inv(mat)
	# def theta(m):
	# 	pass
	# def parameters(A,tol):
	# 	if cond ==True:
	# 		pass
	# def exponent_mat(self,A,B,t):
	# 	# balance # if self.check_balance()==True:

	# 	mu=np.trace(A)/nuser # define
	# 	A-= mu*np.eye(nuser)
	# 	if t*LA.norm(A,1) == 0 :
	# 		m_star,s = 0,1
	# 	else:
	# 		m_star,s=parameters(t*A,tol)# define 
	# 	F=B
	# 	eta=np.exp(t*mu/s)
	# 	for i in range(s):# define 
	# 		c1=LA.norm(B, np.inf)
	# 		for j in range(m_star):# define
	# 			B=(t/(s*j))*A.dot(B)
	# 			c2=LA.norm(B, np.inf)
	# 			F=F+B
	# 			if c1+c2 <=tol*LA.norm(F, np.inf):
	# 				break
	# 			c1=c2
	# 		F=eta*F
	# 		B=F
	# 	if self.check_balance()==True:
	# 		# define
	# 		return D.dot(F)
	# 	return F

		