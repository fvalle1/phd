import pandas as pd
import numpy as np
from numpy.random import Generator, PCG64, SeedSequence
import multiprocessing as mp
import time, sys
import tensorflow as tf
import tensorflow_probability as tfp

class Mazzolini():
	data = []
	def __init__(self, df, ensamble=50, seed=4242):
		self.generators = {}
		self.df = df
		self.M, self.f = self._get_reduce()
		self.ensamble = ensamble
		self.data = []
		self.sg = SeedSequence(seed)
		
	def _get_reduce(self):
		f = self.df.sum(1)
		return (self.df.sum(0), f/f.sum())

	def _func(self, m):
		whoami = mp.current_process().name
		if whoami in self.generators.keys():
			rng = self.generators[whoami]
		else:
			rng = Generator(PCG64(self.sg.spawn(1)[0]))
			self.generators[whoami] = rng
		return np.average([np.random.multinomial(m, self.f.values) for stat in range(self.ensamble)], 0)

	def append(self,d):
		global data
		self.data.append(d)
		
	def call_error(self, err):
		print(*sys.exc_info(), err)

	def run_parallel_async(self, threads=mp.cpu_count()):
		global data
		self.data = []
		pool=mp.Pool(threads)
		results = pool.map_async(self._func,  self.M.values, callback=self.append, error_callback=self.call_error)
		pool.close()
		pool.join()
		return pd.DataFrame(index=self.df.index, columns=self.df.columns, data=np.array(self.data).T.reshape(self.df.shape))


	def run_parallel(self, threads=mp.cpu_count()):
		pool=mp.Pool(threads)
		results = pool.map(self._func,  self.M.values)
		pool.close()
		pool.join()
		return pd.DataFrame(index=self.df.index, columns=self.df.columns, data=np.array(results).T.reshape(self.df.shape))

	def run(self):
		df_mazzolini = pd.DataFrame(index=self.df.index)
		for doc, m in self.M.iteritems():
			df_mazzolini.insert(0,doc, np.average([np.random.multinomial(m,self.f.values) for stat in range(self.ensamble)], 0))
		return df_mazzolini

	def run_tf(self):
		self.data = []
		df_mazzolini = pd.DataFrame(index=self.df.index)
		dist = tfp.distributions.Multinomial(self.M.values.astype(float), probs=self.f.values)
		data = dist.sample().numpy()
		return pd.DataFrame(index=self.df.index, columns=self.df.columns, data=np.array(self.data).T.reshape(self.df.shape))
	