import sys
sys.path.append("/home/jovyan/work/phd/hsbm-occam")
from sbmtm import sbmtm
import graph_tool as gt
import pandas as pd

import unittest

class Testsbmtm(unittest.TestCase):
	def setUp(self, *args):
		self.model=sbmtm()
		self.mat = None

	def test_create(self):
		self.assertIsInstance(self.model,sbmtm)

	def test_get_mat(self):
		model = self.model
		model.load_graph("/home/jovyan/work/phd/datasets/paper/gtex10seed/topsbm/graph.xml.gz")
		self.mat = gt.spectral.adjacency(model.g, model.g.edge_properties["count"])[len(model.documents):,:len(model.documents)]
		self.assertEqual(len(model.words),self.mat.shape[0])

	def test_mat_shape(self):
		self.model.load_graph("/home/jovyan/work/phd/datasets/paper/gtex10seed/topsbm/graph.xml.gz")
		model = self.model
		self.mat = gt.spectral.adjacency(model.g, model.g.edge_properties["count"])[len(model.documents):,:len(model.documents)]
		self.assertEqual(len(self.model.documents), self.mat.shape[1])



if __name__=="__main__":
	unittest.main()
