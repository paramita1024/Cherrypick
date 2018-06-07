from PriorityQueue import PriorityQueue
import math
import numpy.random as rnd
import numpy as np
from numpy import linalg as LA
from scipy import linalg as scipyLA
from a import graph
import heapq
# from collections import deque


class fixed_size_list:
	
class slant:
	def __init__(self,obj):
		self.num_node = obj.num_node
		self.num_sentiment_val = obj.num_sentiment_val # *****************
		self.nodes = obj.nodes
		self.graph=obj.graph
		self.train= obj.train
		self.test = obj.test
		self.num_train= self.train.shape[0]
		self.num_test= self.test.shape[0]
		self.num_simulation = 100000 # 
		self.w = 0 # 
		self.var = 0 # 
		self.v = 0 #
		self.size_of_function_val_list = 1
		# self.MSE
		# alpha
		# A
		# mu
		# B

	def project_positive_quadrant(v_array):
		v_pos = np.array(v_array)
		for i in range(v_array.shape[0]):
			if v_array[i] < 0 :
				v_pos[i] = -v_pos[i] 
		return v_pos
	def find_mu_B(self):
		mu=np.ones(self.num_node)
		B=np.zero((self.num_node,self.num_node))
		for user in self.nodes:
			# self.mu[user], self.B[user,] =self.spectral_proj_grad()
			k=0
			B_user_init = np.ones( self.num_node ) # change
			x=np.concatenate(mu[user], B_user_init, axis=1 )
			func_val_list=[]
			d = np.ones( x.shape[0] ) 
			H = np.eye( x.shape[0] )
			alpha_bb = .0001 +.5*rnd.uniform(0,1) 
			alpha_min = .0001 
			while LA.norm(d) > sys.float_info.epsilon:
				alpha_bar=min([alpha_min,max(alpha_bb, alpha_min)]) 
				# alpha
				d=self.project_positive_quadrant(x-alpha_bar*self.grad_f(x))-x #
				func_val_list.append(self.f(x)) #
				if len(func_val_list) > self.size_of_function_val_list: 
					func_val_list.pop(0)
				max_func_val=max(func_val_list) 
				alpha = 1
				lhs = 1 # check code
				while lhs > max_func_val + self.v*alpha*self.grad_f(x).dot(d): # lhs # write B as H 
					alpha = alpha - .1
				s = alpha * d
				x = x + s
				y = H.dot(d)
				alpha_bb = y.dot(y) / s.dot(y) 
				# Bk=Bk-Bk*(sk*sk')*Bk/(sk'*Bk*sk)+yk*yk'/(yk'*sk);
				H_s = H.dot(s) 
				# B  = B - B @ (nps  s.T ) @ B / (s.dot(B.dot(s))) + np.matmul(y,y.T)
				H = H - s.dot(H_s) * np.matmul( H_s, H_s.T) + np.matmul( y, y.T )/ y.dot(s)
				k=k+1

		self.mu = mu
		self.B = B

		

	def solve_least_square(self,A,b):
		
		x=LA.solve(np.matmul(A.T,A),np.dot(A.T,b))
		return x[0],x[1:]
	# def find_alpha_A(self):
	# 	# maintain msg counter for each user
	# 	# map users to 0 to nusers
	# 	ind_4_V=np.zero(self.nuser)
	# 	tau={}
	# 	for u in range(nuser):
	# 		tau[u]=[]
	# 	for t,u,m in H: #
	# 		tau[u].append([t,u,m])
	# 		ind_4_v[u]=ind_4_v[u]+1
	# 	for u in range(nuser):
	# 		i=0
	# 		S=tau[u]
	# 		for nbr in graph[u]:
	# 			S=self.merge(S,tau[nbr])#
	# 			x_last=0
	# 			t_last=0
	# 			m_no=0
	# 			for t,v,m in S:
	# 				if v==u:
	# 					x=x_last*math.exp(-w*(t-t_last))
	# 					g(m_no,v)=x
	# 					y(m_no)=m
	# 					m_no=m_no+1
	# 				else:
	# 					x=x_last*math.exp(-w*(t-t_last))
	# 				t_last=t
	# 				x_last=x
	# 		alpha[u],A[u]=self.solve_least_square(g,y,lda)
	# 	# nmsg
	# 	# M=np.array(nuser,nmsg.max(),3)
	# 	# for u in self.graph.vertices:
	# 	# 	M[u,:,:1]=np.asarray([train[i,1:] if train[i,0]==u for i in self.ntrain])
	# 	# 	M[u,:,2]=np.multiply(M[u,:,0],np.exp(w*M[u,:,1]))
	# 	# alpha=np.array(nuser)
	# 	# A=[]
	# 	# for u in self.graph.vertices:
	# 	# 	M_hat=np.array(nmsg,len(self.edges[u])+1)
	# 	# 	arg1=M_hat.transpose().dot(M_hat)
	# 	# 	arg2=M_hat.transpose().dot(M[u,:,0])
	# 	# 	res = LA.solve(arg1,arg2)
	# 	# 	alpha[u]=res[0]
	# 	# 	A.append(list(res[1:]))
	# 	# 	alpha[u],A[u]=self.solve_least_square(A,b)
	# 	return alpha,A

	def find_alpha_A(self):
		alpha = np.zero(self.num_node)
		A = np.zero((self.num_node, self.num_node ))
		index={}
		for user in self.nodes : 
			index{user} = np.where(self.train[:,0]==user)

		for user in self.nodes : 
			user_ind = 0
			opn=0
			time=0
			num_msg_user = index{user}.shape[0]
			neighbours = np.nonzero(self.edges[user,:])
			num_nbr = neighbours.shape[0]
			msg_user = np.zero(num_msg_user)
			g_user = np.zero( ( num_msg_user, num_nbr ) )
			for nbr in neighbours :
				nbr_no = np.where(neighbours == nbr) 
				index_for_both = np.sort( np.concatenate(index{user}, index{nbr}, axis = 1 ) )

				for ind in index_for_both : 
					user_curr , time_curr , sentiment = self.train[ind,:]
					if user_curr == user:
						opn = opn*np.exp(-self.w*(time_curr - time))
						msg_user[user_ind] = sentiment
						user_ind = user_ind + 1
					else:
						opn = opn*np.exp(-self.w*(time_curr - time))+sentiment
						g_user[user_ind, nbr_no]=opn
					time = time_curr
			alpha[user], A[user,neighbours] = self.solve_least_square(msg_user, g_user, self.lambda_user)
		self.alpha = alpha
		self.A = A


	def estimate_param(self, evaluate = True):
		# estimate parameters
		if evaluate == True:

			self.mu,self.B = self.find_mu_B()
			self.alpha, self.A = self.find_alpha_A()
		else: # generate parameters for testing prediction part 

			self.mu = rnd.uniform(size = self.nuser)
			self.A = self.B = []
			for user in self.nodes:
				num_nbr = len(self.graph.edges[user])
				A_user = rnd.uniform(size = num_nbr ) # chec whether return ndarray
				B_user = rnd.uniform( size = num_nbr )
				self.A.append(A_user)
				self.B.append(B_user)
			self.alpha = rnd.uniform( low = 0 , high = self.num_sentiment_val, size = self.nuser )
	def predict(self):
		# test is a dictionary
		# test['user']=[set of msg posted by that user]
		# each user is a key 
		# each user has a list of messages attached to it.
		# for each message ,
		# sample a set of msg first
		# predict the msg of that user at that time
		# save the prediction in a dictionary called prediction 
		# return a set of predicted msg
		# add a loop here to run the following simulation repeated times
		self.MSE = np.zero(self.num_simulation)
		for simulation_no in range(self.num_simulation):
			predict_test = self.predict_by_simulation() 
			self.MSE[simulation_no] = self.get_MSE(predict_test, self.test[:,1]) # define performance
			# self.performance.FR = self.get_FR(predict_test, test[:,1]) 
		return np.mean(self.MSE)
	def predict_by_simulation(self):
		t_old = 0# self.test[0,1]
		# discuss the first case
		predict_test = np.zero(self.num_test)
		for m_no in range(self.num_test):
			t_new= self.test[m_no, 1 ]
			user = self.test[m_no, 0 ]
			del_t = t_new-t_old
			msg_set, opn_update, int_update = self.simulate_events(del_t)
			# predict_test[m_no] = self.predict_from_msg_set( user, t_new, msg_set)
			t_last,opn_last = opn_update[user,:]
			predict_test[m_no] = self.find_opn_markov(opn_last, t_new-t_last, self.alpha[user], self.w)
		return predict_test
	

	def find_opn_markov(self, opn_last , del_t, alpha, w):
		return alpha + (opn_last - alpha)*np.exp(- w * del_t) 
	def simulate_events(self,T):#
		# test message set , parameters learnt from train , T  , graph ( number of node and adj list )
		# sample events 
		# sample events for each user
		# until we reach T , we generate min t , generate corresponding event ,  update all neighbours 
		# predict message sentiment for each msg in test set
		# return prediction 

		
		time_init = np.zero((self.num_node,1))
		opn_update = np.concatenate(time_init, self.alpha.T, axis=1)
		int_update =  np.concatenate(time_init, self.mu.T, axis=1)
		
		H=[]
		tQ=np.zero((self.nuser))
		for u in range(nuser):
			tQ[u]=self.sample_event(self.mu[u],0,u) 
		Q=PriorityQueue(tQ)
		t_old = 0
		while t_old < T:

			t_new,u=Q.extract_prior()# do not we need o put back t_new,u 
			# t_old=opn_update_time[u]
			# x_old=opn_update_val[u]
			[t_old,x_old] = opn_update[u,:]
			x_new=self.alpha[u]+(x_old-self.alpha[u])*np.exp(-self.w*(t_new-t_old))
			# opn_update_time[u]=t_new
			# opn_update_val[u]=x_new
			opn_update[u,:]=np.array([t_new,x_new])
			m=rnd.normal(x_new,self.var,1)#**
			H.append(np.array([u,t_new,m]))
			# update neighbours
			for nbr in np.nonzero(self.edges[u,:]):
				# change above 
				[t_old,lda_old] = int_update[nbr,:]
				lda_new = self.mu[nbr]+(lda-self.mu[nbr])*np.exp(-self.w*(t_new-t_old))+self.B[u,nbr]# use sparse matrix
				int_update[nbr]=np.array([t_new,lda_new])#**
				t_old,x_old=opn_update[nbr,:]
				x_new=self.alpha[nbr]+(x_old-self.alpha[nbr])*np.exp(-self.w*(t_new-t_old))+self.A[u,nbr]*m
				opn_update[nbr,:]=np.array([t_new,x_new])
				t_nbr=self.sample_event(lda_new,t_new,nbr)
				self.Q.update_key(t_nbr,nbr) 
			t_old = t_new
		return np.array(H)
	def sample_event(self,lda,t,v): # to be checked
		lda_bar=lda
		t_new=t
		while t_new<self.T: #* get T 
			u=sample_uniform(0,1)#* 
			t_new=t_new-math.log(u)/lda_bar
			lda_new=self.mu[v]+(lda-self.mu[v])*np.exp(-self.w*(t_new-t))
			d=sample_uniform(0,1)# **
			if d*lda_bar<lda:
				return t_new
			else:
				lda_bar=lda_new
		return t_new
	def get_FR(self,s,t):
		return (len(s)-np.count_nonzero(np.sign(s)+np.sign(t)))/len(s)
	def get_MSE(self,s,t):
		return ((s-t)**2).mean()
	# def analytical_opinion_forecast_poisson(self,train, test): # analytical using poisson
	# 	# create nbr[u]
	# 	time = train[:,2] - t0 
	# 	temp = np.exp(-time*w).dot(train[:,1])
	# 	sentiment =np.array(nuser)
	# 	for u in range(nuser):
	# 		sentiment[u]=sum(np.asarray([temp[i] if train[i,0]==u for i in range(ntrain)]))
		
	# 	x_t0=self.alpha[u]+A[u,self.graph.edges[u]].dot(sentiment[self.graph.edges[u]])
	# 	mat_arg=self.A.dot(np.diag(self.mu))-w*np.eye(nuser)
	# 	inv_mat = self.inverse_mat(mat_arg)
	# 	res=np.zeros(ntest)
	# 	for m in range(ntest):
	# 		exp_mat = scipyLA.expm(self.delta_t*mat_arg)
	# 		sub_term2 = (exp_mat-eye(nuser))*self.alpha
	# 		res[m] = exp_mat[u]*x_t0+w[u]*inv_mat[u]*sub_term2
	# 	return res		
def main():
	
	# input_file = 'input_data'
	# # load data 
	# with open(input_file+'.pkl','rb') as f:
	# 	data = pickle.load(f)
		# data is object with three component
		# graph , 
		# train ( dictionary of msg )
		# test ( dictionary of msg )
	# slant_obj = slant(data.graph)
	# slant_obj.estimate_param(data.train)
	# result = slant_obj.predict(data.test)
	#******************** test *******************************************
	# print type(rnd.uniform(size = num_nbr ))
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

		
