# import matplotlib.pyplot as plt
# from ctypes import *
import numpy as np 
import numpy.random as rnd
import pickle
from math import floor
class test:
	def __init__(self,a,b,c):
		self.a = a 
		self.b = b
		self.c = c
	def print_it(self):
		print "check"
		print "a"
		print self.a
		print "b"
		print self.b
		print "c" 
		print self.c 
# def test_it( num_var ):
# 	s = np.array([3 , 4, 5])
# 	t = np.array([6,7,8])
# 	u = np.array([1,2,3])
# 	if num_var == 1:
# 		return s 
# 	if num_var == 2:
# 		return s, t 
# 	if num_var == 3 : 
# 		return s,t,u
def test_least_square():
	d0 = 30
	d1 = 10
	var = .00001 
	A = rnd.rand( d0, d1)
	x = rnd.rand( d1)
	# print 
	error = rnd.normal( 0 , var , d0 )
	# print error.shape
	# error = error.reshape(error.shape[0],1)
	# print error.shape
	# # print type(error)
	b = A.dot(x) + error
	print b.shape
	# error_loss = []
	# error_in_x = []
	# for self.lambda_least_square in [ 10^x for x in range(-5,5,1) ]:
	# 	# x_computed = self.solve_least_square(A,b)
	# 	x_computed = LA.lstsq(A,b)[0]
	# 	print x_computed.shape
	# 	# print x_computed
	# 	# error_loss.append( LA.norm(A.dot(x_computed) - b ))
	# 	error_in_x.append(LA.norm( x - x_computed ))

	# 	# plt.plot(error_loss)
	# 	# # plt.ylabel('some numbers')
	# 	# plt.show()
	# # print "error in Ax-b"
	# # print error_loss
	# print "error in x - x*"
	# print error_in_x

def main():
	print np.linalg.norm(np.array([1,2,3]))
	# print np.arange(10)
	# test_least_square()
	# # A = np.array([[1,2,3],[4,5,6]])
	# A = np.random.rand(3,3)
	# b=np.array([1,2,3])
	# e= np.random.normal(size=3)
	# print e

	# alpha = np.array([1,2,3])
	# mu = np.array([4,5,6])
	# time_init = np.zeros((3 ,1))

	# opn_update = np.concatenate((time_init, alpha.reshape( 3 , 1 )), axis=1)
	# int_update =  np.concatenate((time_init, mu.reshape( 3 , 1 )), axis=1)
	# # print ctypes.addressof(opn_update)
	# print "---------------opn----------------"
	# print opn_update
	# print "---------------int----------------"
	# print int_update

	# time_init[0] = 35
	# print "---------------opn-----------------"

	# print opn_update
	# print "---------------int-----------------"
	# print int_update

	# print opn_update 

	# print int_update
	# num_var = 3 
	# u = test_it(num_var)
	# print u
	# arr = np.array([1,2,3])
	# print arr
	# test_it(arr)
	# print arr
	# print np.matmul(arr.reshape(arr.shape[0],1) , arr.reshape(1,arr.shape[0]))
	# arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
	# arr_part = arr[[0,2],:]
	# arr[0,0] = 35
	# print arr
	# print arr_part
	# arr = np.array([1,2,3])
	# arr1 = np.array([3,4,5])
	# print np.multiply(arr, arr1)
	# print arr*arr1

	# arr = np.array([1,2,3])
	# print np.concatenate(([4],arr))
	# arr = np.array([[1,2,3],[4,5,6]])
	# for a,b,c in arr:
	# 	print a 
	# 	print b 
	# 	print c
	# v = np.array([1,2,3])
	# print v
	# print v.T
	# print v.shape

	# v1=np.array([v])
	# print v1
	# print v1.T
	# print v1.shape
	# v1 = np.array([[1,0], [0,1]])
	# v2 = np.array([[0],[1]])
	# # v3 = v2.T 
	# v3 = v1.dot(v2)
	# print v3.shape
	# print np.matmul(v1,v2)
	# g=v2.shape
	# print type(g)
	# v = np.array([1,-1,2])
	# v1 = project_positive_quadrant(v)
	# print v1
	# print v
	# t=test(1,2,3)
	# with open('test_file'+'.pkl','wb') as f:
	# 	pickle.dump(t, f, pickle.HIGHEST_PROTOCOL)
	# del t
	# try:
	# 	print t
	# except NameError:
	# 	print "t does not exist"
	# else:
	# 	print " t exists"
	# with open('test_file'+'.pkl','rb') as f:
	# 	t_new = pickle.load(f)
	# t_new.print_it()
if __name__=="__main__":
	main()







# class tree_op:
# 	def __init__(self):
# 		return
# 	def parent(self,i):
# 		if i==0:
# 			print('given index is root')
# 			return -1
# 		return int(floor((i-1)/2))
# 	def left(self,i):
# 		return 2*i+1
# 	def right(self,i):
# 		return 2*i+2


# class Qarray:
# 	def __init__(self,Q):
# 		n=len(Q)
# 		self.Q=np.zeros((n,2))		
# 		for u in range(n):
# 			self.Q[u]=[Q[u],u]
# 	def cmp(self,i,j):
# 		#print type(i)
# 		#print type(j)
# 		return (self.Q[i][0] < self.Q[j][0])
# 	def cmp_with_val(self,key,i):
# 		return ( key < self.Q[i][0] )
# 	def exchange(self,i,j):
# 		temp = np.array(self.Q[i])
# 		self.Q[i]=self.Q[j]
# 		self.Q[j]=temp
# 	def get(self,i):
# 		return self.Q[i]
# 	def update_key(self,i,key):
# 		self.Q[i][0]=key
# 	def check_eq(self,i,j):
# 		return self.Q[i][0]==self.Q[j][0]
# 	def size(self):
# 	 	return self.Q.shape
# 	def get_index_and_val(self,u,stop_index):
# 		for i in range(stop_index):
# 			if self.Q[i][1]==u:
# 				return i,self.Q[i][0]
# 		print "user not found in heap"
# 		return -1,0
# 	def set(self,i,v): 
# 		self.Q[i]=v
		
# class PriorityQueue(tree_op):
# 	def __init__(self,Q):
# 		tree_op.__init__(self)
# 		self.Q=Qarray(Q)
# 		self.flag_user=np.ones(len(Q))
# 		self.heapsize=len(Q)
# 		self.max_size = self.heapsize
# 		self.build_heap()
# 	def build_heap(self):
# 		for i in range(int(floor(self.heapsize/2))-1,-1,-1):
# 			self.heapify(i)
# 	def heapify(self,i):
# 		l=self.left(i) 
# 		r=self.right(i)  
# 		selected=i # selected is the index to be heapified next time
# 		if l<self.heapsize:
# 			if self.Q.cmp(l,selected): 
# 				selected=l
# 		if r<self.heapsize:
# 			if self.Q.cmp(r,selected): 
# 				selected=r
# 		if selected != i:
# 			self.Q.exchange(i,selected) 
# 			self.heapify(selected)
# 	def insert(self,t): 
# 		self.heapsize=self.heapsize+1
# 		if self.heapsize <= self.max_size:
# 			self.Q.set(self.heapsize-1,[float('Inf'),t[1]])
# 			self.minheap_dec_key(self.heapsize-1,t[0])
# 		else:
# 			print "maximum size of heap is reached" 
# 	def extract_prior(self):
# 		if self.heapsize < 1 :
# 			print("heap underflow")
# 		val=np.array(self.Q.get(0))
# 		#print "val"

# 		self.Q.exchange(0,self.heapsize-1)
# 		self.heapsize=self.heapsize-1
# 		self.heapify(0)
# 		self.flag_user[int(val[1])]=0
# 		return val
# 	def minheap_inc_key(self,i,key):
# 		if self.Q.cmp_with_val(key,i): 
# 			print("new key is smaller than current key")
# 		self.Q.update_key(i,key)
# 		self.heapify(i) 
# 	def minheap_dec_key(self,i,key):		
# 		if ~self.Q.cmp_with_val(key,i):
# 			print("new key is not smaller than current key")
# 		self.Q.update_key(i,key)
# 		while i>0:
# 			p=self.parent(i)	
# 			if self.Q.cmp(i,p): 
# 				self.Q.exchange(i,p)
# 				i=p
# 			else:
# 				break
# 		return
		
# 	def update_key(self,t,u):
# 		if self.flag_user[u]==0:
# 			self.flag_user[u]=1
# 			self.insert(np.array([t,u]))
# 		else:
# 			ind,old_t=self.Q.get_index_and_val(u,self.heapsize)
# 			if t > old_t:
# 				self.minheap_inc_key(ind,t)
# 			else:
# 				self.minheap_dec_key(ind,t)
# 	def print_heap(self):
# 		i=0
# 		c=1
# 		while i<self.heapsize:
# 			# for j in range(i,i+c):
# 			# 	print self.Q.Q[j][0]
# 			# print "----------"
# 			if i+c<=self.heapsize:
# 				print self.Q.Q[i:i+c,0]
# 			else:
# 				print self.Q.Q[i:self.heapsize,0]
# 			i=i+c
# 			c=c*2

# Q_init=[50,32,2,21,0,4,1,3,5,6,7]
# # Q=Qarray(Q_init)
# #print Q.Q
# # print Q.cmp(1,2)

# Q=PriorityQueue(Q_init)
# print Q.Q.Q#print_heap()
# #print Q.heapsize
# print Q.extract_prior()
# #print Q.heapsize
# #print Q.flag_user
# Q.update_key(40,5)
# print Q.Q.Q
# #print Q.heapsize
# #print Q.flag_user
# #Q.print_heap()
# # print Q.Q.Q
# # print Q.Q.get_index_and_val(3,Q.heapsize)
# # print "over"
# # Q.minheap_inc_key(2,6)
# # print "updated"
# # Q.print_heap()
# # Q.minheap_dec_key(5,-.25)
# # print "updated"
# # Q.print_heap()


# #a=Q.extract_prior()
# #Q.update_key(7,4) 
# #print a
# #print Q.flag_user
# #Q.print_heap()
# # print range(3,-1)
# # Q=[41,32,65]
# # Q_new=Qarray(Q)
# # print Q_new.check_eq(0,1)
# # Q_new.update_key(1,41)
# # print Q_new.check_eq(0,1)
# # print Q_new.Q.size
# # for i in range(len(Q)):
# # print Q_new.Q
# # print Q_new.check_eq(0,1)
# # Q_new.set(1,np.array([41,1]))
# # print Q_new.Q
# # print size(Q_new.Q)
# # Q_new.check_eq(0,1)
# # Q_new.update_key(1,34)
# # print Q_new.Q

# #print type(Q_new.Q)
# #Q_new.exchange(0,2)
# #print type(Q_new.get(1))
# # t=tree_op()
# # print t.right(2)