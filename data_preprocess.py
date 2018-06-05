class data:
	def __init__(self):
		self.graph = graph
		self.train = train
		self.test = test
		self.nuser = graph.get_nuser() # to be defined
class data_preprocess:
	def __init__(self,file):
		self.file = file
	def read_data(self):
		pass
	def save_data(self,obj_file):
		data_obj = data(graph , train, test )
		with open(obj_file+'.pkl','wb') as f:
			pickle.dump(data_obj, f, pickle.HIGHEST_PROTOCOL)
def main():
	file = 'a'
	fraction_of_test = .1
	dp = data_preprocess(file)
	dp.read_data()
	dp.split_data(frac = fraction_of_test)
	dp.save_data()
if __name__ == "__main__":
	main()

