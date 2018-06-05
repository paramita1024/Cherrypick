import numpy as np
import numpy.random as rnd
from math import ceil
class create_synthetic_data:
	def __init__(self, num_node, frac_sparse, num_msg, num_sentiment_val, start_time, end_time):
		self.num_node = num_node
		self.nodes = range(self.num_node)
		self.frac_sparse = frac_sparse
		self.num_msg = num_msg
		self.num_sentiment_val = num_sentiment_val
		self.start_time = start_time
		self.end_time = end_time
	def generate_graph(self):
		self.edges={}
		shuffled_items = list(self.nodes)
		for i in self.nodes:
			#self.edges[i]=list(rnd.choices(self.num_node, int( self.frac_sparse * self.num_node ) ))
			rnd.shuffle(shuffled_items)
			#print shuffled_items
			self.edges[i]=list(shuffled_items[:int( self.frac_sparse * self.num_node )])
			if i in self.edges[i]:
				self.edges[i].remove(i)
	def generate_msg(self):
		self.msg={}
		for i in self.nodes:
			self.msg[i] = []
		time_fraction = ( self.end_time - self.start_time)/self.num_msg
		start_time= self.start_time
		end_time = start_time + time_fraction

		for m_no in range(self.num_msg):
			user = rnd.randint(0,self.num_node)
			time= rnd.uniform(start_time,end_time)
			sentiment = rnd.randint(0,self.num_sentiment_val)
			#print "u: " + str(user) + " time: " + str(time) + " sentiment: " + str(sentiment)
			self.msg[user].append([time,sentiment])
			start_time = end_time
			end_time = start_time + time_fraction
	def split_data(self, train_fraction):
		self.train ={}
		self.test = {}
		for u in self.nodes:
			num_msg = len(self.msg[u])
			num_tr = int(num_msg*train_fraction)
			self.train[u]=self.msg[u][:num_tr]
			self.test[u] = self.msg[u][num_tr:]
			# print "-----------------------user "+str(u)+"----------------"
			# for msg in self.msg[u]:
			# 	print msg
			# print "-------------------------------------------------------"
			# for msg in self.train[u]:
			# 	print msg
			# print "--------------------------------------------------------"
			# for msg in self.test[u]:
			# 	print msg
def main():
	# define parameters
	# train_fraction = .9
	num_node = 4
	frac_sparse = .4
	num_msg = 40
	num_sentiment_val = 5
	start_time = 0
	end_time = 40
	data = create_synthetic_data(num_node, frac_sparse, num_msg, num_sentiment_val, start_time, end_time)
	#a = [0,1,2,3,4,5]
	#rnd.shuffle(a)
	#print type(ceil(1.5))
	data.generate_graph()
	#print data.nodes
	#print data.num_node
	#print data.msg
	data.generate_msg()
	data.split_data(.6)
	# for i in range(20):
	# 	print rnd.randint(1,5)
	# print data.msg
	# data.split_data(train_fraction)
	output_file = "synthetic_data_1"
	with open(output_file+'.pkl' , 'wb') as f:
		save( data, f , pickle.HIGHEST_PROTOCOL)
if __name__=="__main__":
	main()