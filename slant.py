import matplotlib.pyplot as plt
from create_synthetic_data import create_synthetic_data
import pickle
from PriorityQueue import PriorityQueue
import math
import numpy.random as rnd
import sys
import numpy as np
from numpy import linalg as LA
# from scipy import linalg as scipyLA
# from a import graph
# import heapq
# from collections import deque
	
class slant:
	def __init__(self,obj):
		self.num_node = obj.num_node
		self.num_sentiment_val = obj.num_sentiment_val # *****************
		self.nodes = obj.nodes
		self.edges=obj.edges
		self.train= obj.train
		self.test = obj.test
		self.num_train= self.train.shape[0]
		self.num_test= self.test.shape[0]
		self.num_simulation = 1 # 00000 # 
		self.w = 1 # 
		self.var = 1 # 
		self.v = 1 #
		self.lambda_least_square = 1
		self.spectral_nu = .8
		# self.size_of_function_val_list = 1
		# self.MSE
		# alpha
		# A
		# mu
		# B

	def project_positive(self,v):
		return np.maximum(v,np.zeros(v.shape[0]))

	def Grad_f_n_f(self, coef_mat_user, last_coef_val, num_msg_user, msg_time_exp_user,  user_mask, x):
		last_time_train = self.train[-1,1] 
		mu = x[0]
		b = x[1:]
		common_term = np.reciprocal(coef_mat_user.dot(b) * msg_time_exp_user + mu)
		del_b_t1 = coef_mat_user.T.dot( common_term *  msg_time_exp_user)
		del_mu_t1= np.sum(common_term)
		del_b_t2 = (np.exp(self.v*last_time_train)*(last_coef_val*user_mask) - num_msg_user*user_mask ) / self.v
		del_mu_t2= last_time_train
		del_t1 = np.concatenate( ([del_mu_t1], del_b_t1))
		del_t2 = np.concatenate( ([del_mu_t2], del_b_t2))
		# function value computation 
		t1 = np.sum( np.log( coef_mat_user.dot(b) * msg_time_exp_user + mu ) )
		t2 = mu + (  np.exp(self.v*last_time_train)*(b.dot(last_coef_val*user_mask)) - b.dot(num_msg_user*user_mask)) / self.v

		grad_f =del_t1 - del_t2
		function_val = t1 - t2 
		return grad_f , function_val
	def find_mu_B(self):
		mu=np.ones(self.num_node)
		B=np.zeros((self.num_node,self.num_node))

		#  gradient calculation
		coef_mat = np.zeros((self.num_train, self.num_node))
		last_coef_val = np.zeros( self.num_node)
		num_msg_user = np.zeros(self.num_node)
		msg_time_exp = np.zeros(self.num_train)
		msg_index = 0 
		for user , time , sentiment  in self.train:
			user = int(user)
			neighbours_with_user  = np.concatenate(([user], np.nonzero(self.edges)[0] ))
			value = np.exp(-self.v * time)
			msg_time_exp[msg_index] = 1/value
			print user
			last_coef_val[user] = last_coef_val[user] + value
			coef_mat[msg_index,neighbours_with_user] = last_coef_val[neighbours_with_user]
			num_msg_user[user] = num_msg_user[user] + 1
			msg_index = msg_index + 1 
		
		for user in self.nodes:
			# self.mu[user], self.B[user,] =self.spectral_proj_grad()
			B_user_init = np.ones( self.num_node ) # change
			x=np.concatenate(([mu[user]], B_user_init))
			# func_val_list=[]
			d = np.ones( x.shape[0] ) 
			H = np.eye( x.shape[0] ) 
			alpha_bb = min([.0001 +.5*rnd.uniform(0,1) , 1 ])
			alpha_min = .0001
			# compute parameters***************  
			user_msg_index = np.where( self.train[:,0]==user )[0]
			coef_mat_user =  coef_mat[user_msg_index,:]
			msg_time_exp_user = msg_time_exp[user_msg_index]
			user_mask = self.edges[user,:]
			user_mask[user] = 1
		
			
			while LA.norm(d) > sys.float_info.epsilon:

				grad_f , likelihood_val = self.Grad_f_n_f(coef_mat_user, last_coef_val, num_msg_user ,  msg_time_exp_user, user_mask, x)
				alpha_bar=min([alpha_min,max(alpha_bb, alpha_min)]) 
				# grad_f = np.ones(self.num_node+1) #**
			# 	# alpha
				d=self.project_positive(x-alpha_bar*grad_f)-x #
			# 	# func_val_list.append(self.f(x)) #
			# 	# if len(func_val_list) > self.size_of_function_val_list: 
			# 		# func_val_list.pop(0)
			# 	# max_func_val=max(func_val_list) 
				alpha = 1
				while ( alpha*d.dot(H.dot(d)) > (self.spectral_nu - 1 )*grad_f.dot(d) ) & (alpha > 0) : # lhs # write B as H 
					alpha = alpha - .1
					print "alpha is " + str(alpha)
				print alpha
				
				s = alpha * d
				x = x + s


				y = H.dot(d)
				alpha_bb = y.dot(y) / s.dot(y) 
				# Bk=Bk-Bk*(sk*sk')*Bk/(sk'*Bk*sk)+yk*yk'/(yk'*sk);
				H_s = H.dot(s).reshape(H.dot(s).shape[0],1)
				y_2dim = y.reshape(y.shape[0],1)
				H = H - s.dot(H.dot(s)) * np.matmul( H_s , H_s.T ) + np.matmul( y_2dim , y_2dim.T )/ y.dot(s)
				
				break # to be deleted
				

		self.mu = mu
		self.B = B
		 

	def test_least_square(self):
		d0 = 30
		d1 = 10
		var = .00001 
		A = rnd.rand( d0, d1)
		x = rnd.rand( d1)
		# print x.shape
		error = rnd.normal( 0 , var , d0 )
		# print error
		# error = error.reshape(error.shape[0],1)
		# print error.shape
		# print type(error)
		b = A.dot(x) + error
		# print A.dot(x).shape
		error_loss = []
		error_in_x = []
		x_axis = np.arange(d1)
		# print x_axis.shape
		for self.lambda_least_square in [ 10^x for x in range(-5,5,1) ]:
			x_computed = self.solve_least_square(A,b)
			# x_computed = LA.lstsq(A,b)
			# print x_computed.shape
			# print x_computed
			# error_loss.append( LA.norm(A.dot(x_computed) - b ))
			error_in_x.append(LA.norm( x - x_computed ))
			# plt.plot(error_in_x)
			# print x_axis.shape
			# print x.shape
			# print x_computed.shape
			plt.plot(x_axis, x, 'r--', x_axis, x_computed, 'bs')
			plt.show()
			

			# plt.plot(error_loss)
			# # plt.ylabel('some numbers')
			# plt.show()
		# print "error in Ax-b"
		# print error_loss
		# print "error in x - x*"
		# print error_in_x
		plt.plot(error_in_x)
		plt.show()
		

	def solve_least_square(self,A,b):
		
		# print "A"+ str(A.shape)
		# print b.shape
		A_T_b = A.T.dot(b)
		# print A_T_b.shape
		# x=LA.solve( np.matmul(A.T,A)+self.lambda_least_square*np.eye( A_T_b.shape[0] ) , A.T.dot(b))
		x=LA.solve( np.matmul(A.T,A), A.T.dot(b))
		print x.shape
		return x #**************
		# return x[0],x[1:]
	

	def find_alpha_A(self):
		
		alpha = np.zeros(self.num_node)
		A = np.zeros((self.num_node, self.num_node ))
		index={}
		for user in self.nodes : 
			index[user] = np.where(self.train[:,0]==user)[0]
		# print self.train[:,0]
		# print "_-----------------------------------------------"
		# print index
		# print "nodes"
		# print self.nodes
		# print "_____________________________"
		for user in self.nodes :  
			# user_ind = 0
			opn=0
			time=0
			num_msg_user = index[user].shape[0]
			neighbours = np.nonzero(self.edges[user,:])[0]
			num_nbr = neighbours.shape[0]
			msg_user = self.train[index[user],2]
			g_user = np.zeros( ( num_msg_user, num_nbr ) )
			nbr_no=0
			for nbr in neighbours :
				user_msg_ind = 0
				index_for_both = np.sort( np.concatenate((index[user], index[nbr])) )
				# print "-----------------------------------"
				# print "user" + str(user) 
				# print index[user]
				# print "---"
				# print " nbr : " + str(nbr)
				# print index[nbr]
				print "---------merge------------"
				print index_for_both
				print "______________________________________"
				for ind in index_for_both : 
					user_curr , time_curr , sentiment = self.train[ind,:]
					print "---------msg-----------"
					print self.train[ind,:]

					if user_curr == user:
						opn = opn*np.exp(-self.w*(time_curr - time))
						user_msg_ind = user_msg_ind + 1
						if user_msg_ind == num_msg_user:
							break
					else:
						opn = opn*np.exp(-self.w*(time_curr - time))+sentiment
						g_user[user_msg_ind, nbr_no]=opn
					time = time_curr
				nbr_no = nbr_no + 1
			print "-----------msg_user-----------"
			print msg_user
			print "----------g_user ______________"
			print g_user


			alpha[user], A[user,neighbours] = self.solve_least_square(np.concatenate(( np.ones((msg_user.shape[0],1)) , g_user  ), axis = 1 ), msg_user )
		self.alpha = alpha
		self.A = A


	def estimate_param(self, evaluate = True):
		# estimate parameters
		if evaluate == True:
			# print "inside"
			# self.mu,self.B = self.find_mu_B()
			self.find_mu_B()
			# self.find_alpha_A()
		else: # generate parameters for testing prediction part 

			self.mu = rnd.uniform(low = 0 , high = .05 , size = self.num_node)
			self.alpha = rnd.uniform( low = 0 , high = self.num_sentiment_val-1 , size = self.num_node )
			self.A = np.zeros(( self.num_node , self.num_node ))
			self.B = np.zeros(( self.num_node , self.num_node ))
			for user in self.nodes:
				nbr = np.nonzero(self.edges[user,:])[0]
				# print "________________"
				# print nbr
				# print "___________________"
				self.A[user ,  nbr ] = rnd.uniform( low = -1 , high = 1 , size = nbr.shape[0] ) 
				# print self.A[user , :]
				# print "_____________________"
				self.B[user , np.concatenate(([user], nbr ))] = rnd.uniform( size = nbr.shape[0] + 1  )

			# print self.mu
			# print self.alpha
			# print "____________E__________________________________________--______"
			# print self.edges
			# print "----------A-------------------------------------------------------"
			# print self.A
			# print "___________B___________________________________________________--------___------"
			# print self.B
			# print "----------nodes-------------"
			# print self.nodes
	def get_FR(self,s,t):
		# print float(s.shape[0] - np.count_nonzero(np.sign(s) + np.sign(t))) / s.shape[0]
		return float(s.shape[0] - np.count_nonzero( np.sign(s) + np.sign(t)))/s.shape[0]
	def get_MSE(self,s,t):
		return np.mean((s-t)**2)
		
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
		# v1 = np.array([1,2,2])
		# v2 = np.array([4,5,6])
		# print self.get_MSE(v1,v2)



		self.MSE = np.zeros(self.num_simulation)
		for simulation_no in range(self.num_simulation):
			# predict_test =  rnd.uniform( low = 0 , high = self.num_sentiment_val-1  , size = self.num_test )# 
			predict_test =  self.predict_by_simulation() 
			self.MSE[simulation_no] = self.get_MSE(predict_test, self.test[:,2]) # define performance
			# self.performance.FR = self.get_FR(predict_test, test[:,1]) 
		# print np.mean(self.MSE)
		return np.mean(self.MSE)
	def predict_by_simulation(self):
		t_old= self.train[-1,1]
		# discuss the first case
		predict_test = np.zeros(self.num_test)
		msg_no = 0 
		# for user, time, sentiment in self.test:
		for user, time, sentiment in self.test:
			user = int(user)
			del_t = time-t_old 
			# print user
			# print del_t
			opn_update = self.simulate_events(del_t, return_only_opinion_updates = True) # may send user opn update also # add t_old with time array #************
			# self.simulate_events(del_t, return_only_opinion_updates = True) # may send user opn update also # add t_old with time array
			#-------------------------------------------------------------------
			# opn_update = np.zeros((self.num_node , 2))
			# opn_update[:,0] =  rnd.uniform( low = 0 , high = del_t, size = self.num_node )
			# opn_update[:,1] = rnd.uniform( low = 0 , high = self.num_sentiment_val-1 , size = self.num_node )
			# #---------------------------------------------------------------------
			# print opn_update
			# predict_test[m_no] = self.predict_from_msg_set( user, t_new, msg_set)
			time_diff,opn_last = opn_update[user,:]
			predict_test[msg_no] = self.find_opn_markov(opn_last, del_t - time_diff, self.alpha[user], self.w)
			t_old = time
			msg_no = msg_no + 1 
			break #--------------------------------------------------------------------------
		return predict_test
	

	def find_opn_markov(self, opn_last , del_t, alpha, w):
		# print "-----------------------------------------------------------------------------"
		# print opn_last 
		# print del_t 
		# print alpha 
		# print w 
		# print w * del_t
		# print np.exp(- w * del_t) 
		# print (opn_last - alpha)*np.exp(- w * del_t) 
		# print alpha + (opn_last - alpha)*np.exp(- w * del_t) 
		return alpha + (opn_last - alpha)*np.exp(- w * del_t) 
	def simulate_events(self,T, return_only_opinion_updates = False):#
		# test message set , parameters learnt from train , T  , graph ( number of node and adj list )
		# sample events 
		# sample events for each user
		# until we reach T , we generate min t , generate corresponding event ,  update all neighbours 
		# predict message sentiment for each msg in test set
		# return prediction 


		#________________________checking sample events --------------------------------------------

		# self.sample_event(lda_init = .3, t_init =1 , v =1 , T = 5)

		# ------------------------------------------------------------------------------------------
		
		time_init = np.zeros((self.num_node,1))
		opn_update = np.concatenate((time_init, self.alpha.reshape(self.num_node , 1 )), axis=1)
		int_update =  np.concatenate((time_init, self.mu.reshape( self.num_node , 1 )), axis=1)
		
		msg_set = []

		tQ=np.zeros(self.num_node)
		for user in self.nodes:
			tQ[user] = self.sample_event( self.mu[user] , 0 , user, T ) 
			# tQ[user] = rnd.uniform(0,T)
		Q=PriorityQueue(tQ)
		# print "----------------------------------------"
		# print "sample event starts"
		t_new = 0
		# num_msg = 0
		while t_new < T:

			t_new,u=Q.extract_prior()# do not we need o put back t_new,u 
			u = int(u)

			# print " extracted user " + str(u) + "---------------time : " + str(t_new)
			# t_old=opn_update_time[u]
			# x_old=opn_update_val[u]
			[t_old,x_old] = opn_update[u,:]
			x_new=self.alpha[u]+(x_old-self.alpha[u])*np.exp(-self.w*(t_new-t_old))
			# opn_update_time[u]=t_new
			# opn_update_val[u]=x_new
			opn_update[u,:]=np.array([t_new,x_new])
			m = rnd.normal(x_new,self.var)
			msg_set.append(np.array([u,t_new,m]))
			# num_msg = num_msg + 1 #***
			# if num_msg > 20 :
				# break
			# update neighbours
			for nbr in np.nonzero(self.edges[u,:])[0]:
				# print " ------------for nbr " + str(nbr) + "-------------------------"
				# change above 
				[t_old,lda_old] = int_update[nbr,:]
				lda_new = self.mu[nbr]+(lda_old-self.mu[nbr])*np.exp(self.v*(t_new-t_old))+self.B[u,nbr]# use sparse matrix
				int_update[nbr,:]=np.array([t_new,lda_new])
				t_old,x_old=opn_update[nbr,:]
				x_new=self.alpha[nbr]+(x_old-self.alpha[nbr])*np.exp(-self.w*(t_new-t_old))+self.A[u,nbr]*m
				opn_update[nbr,:]=np.array([t_new,x_new])

				# print " updated int " + str(lda_new) + " ------------ updated opinion -----" + str(x_new)
				t_nbr=self.sample_event(lda_new,t_new,nbr, T )
				# print " update next event time of " + str( nbr ) + "  as " + str(t_nbr)

				Q.update_key(t_nbr,nbr) 
			# Q.print_elements()
			
		if return_only_opinion_updates == True:
			return opn_update
		else:
			return opn_update, int_update, np.array(H) 
	def sample_event(self,lda_init,t_init,v, T ): # to be checked
		lda_upper=lda_init
		t_new = t_init
		 
		
		# print "------------------------"
		# print "start tm "+str(t_init) + " --- int --- " + str(lda_init)
		# print "------------start--------"
		while t_new < T : #* get T 
			u=rnd.uniform(0,1)
			t_new = t_new - math.log(u)/lda_upper
			# print "new time ------ " + str(t_new)
			lda_new = self.mu[v] + (lda_init-self.mu[v])*np.exp(-self.w*(t_new-t_init))
			# print "new int  ------- " + str(lda_new)
			d = rnd.uniform(0.9,1)
			# print "d*upper_lda : " + str(d*lda_upper)
			if d*lda_upper < lda_new  :
				break
			else:
				lda_upper = lda_new
		return t_new
	
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
	input_file = 'synthetic_data_1'
	# load data 
	with open(input_file+'.pkl','rb') as f:
		data = pickle.load(f)
		# data is object with three component
		# graph , 
		# train ( dictionary of msg )
		# test ( dictionary of msg )
	slant_obj = slant(data)
	slant_obj.test_least_square()
	# slant_obj.estimate_param( evaluate = False )
	# slant_obj.predict()
	# result = slant_obj.predict()
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
if __name__=="__main__":
	main()

		
# def find_alpha_A(self):
	# 	# maintain msg counter for each user
	# 	# map users to 0 to nusers
	# 	ind_4_V=np.zeros(self.nuser)
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