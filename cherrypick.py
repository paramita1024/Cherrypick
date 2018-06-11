import numpy as np
from slant.py import slant
class performance:
	def __init__(self):
		self.MSE=0
		self.FR=0	

class cherrypick:
	def __init__(self,filename):
		# read the data as a list of (user,msg,time)
		# split to create per user sorted list of msg
		# split each user list in test and train

		with open(filename,'rb') as f:
			data = pickle.load(f)
			# data is a class containing graph, test and train
		self.graph = data.graph
		self.train = data.train
		self.test = data.test
	def find_H_and_O(self):
		# init H,V,O,I
		H=[]
		V=range(ntrain)
		O=[]
		I=range(nuser)
		# number of user not exceeded, 
		while len(O) <= self.max_end_user:
			# select msg and user 
			H.append(m)
			V.remove(m)
			O.append(u)
			I.remove(u)

		
		# while msg limit has not reached
		# select a msg 
		# include in H , exclude from V
		# return H,O
		while len(H) <= self.max_end_msg:
			# select m
			H.append(m)
			V.remove(m)
		return H,V,O,I
	def train_model(self):
		H,V,O,I = self.find_H_and_O()
		# modify input train test graph
		self.train,self.test, self.train_ex, self.test_ex = self.reduce()
		self.slant_opt= slant(self.graph)
		self.slant_opt.estimate_param(self.train)# define and pass parameters	
	def forecast(self):
		self.result = self.slant_opt.predict_sentiment(self.test)
	# def create_graph(self):
	# 	self.graph={}
	# 	for v in num_v:
	# 		self.graph[v]=set([])
	# 	for node1,node2 in set_of_egdes:
	# 		self.graph[node1].add(node2)
	# def load(self):
	# 	data=np.genfromtxt(self.fp,delimiter=',')
	# 	user,index,count = np.unique(data,return_index=True, return_counts=True)
	# 	for i in range(nuser):
	# 		tr_idx = np.concatenate([tr_idx, index[i,:np.floor(self.split_ratio*count[i])]])
	# 		te_idx = np.concatenate([tr_idx, index[i,np.floor(self.split_ratio*count[i]):]])
	# 	train=data[tr_idx,:]
	# 	test=data[te_idx,:]
	# 	self.ntrain=train.shape[0]
	# 	self.ntest=test.shape[0]
	# 	self.nuser=user.shape[0]

def main():
	filename='yet_not_decided'
  	cherryp = cherrypick(filename)
  	cherryp.train_model()
  	cherryp.forecast()
if __name__== "__main__":
  main()
