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
		model.load_graph("/home/jovyan/work/phd/datasets/gtex/10/topsbm/graph.xml.gz")
		self.mat = gt.spectral.adjacency(model.g, model.g.edge_properties["count"])[len(model.documents):,:len(model.documents)]
		self.assertEqual(len(model.words),self.mat.shape[0])

	def test_mat_shape(self):
		self.model.load_graph("/home/jovyan/work/phd/datasets/gtex/10/topsbm/graph.xml.gz")
		model = self.model
		self.mat = gt.spectral.adjacency(model.g, model.g.edge_properties["count"])[len(model.documents):,:len(model.documents)]
		self.assertEqual(len(self.model.documents), self.mat.shape[1])

	def test_approx(self):
		import pandas as pd
		model = self.model
		self.model.make_graph_from_BoW_df(pd.DataFrame(data=[[1.1,2.2,3.3],[0.5,0.2,0.1],[0.03,0.01,1.005]], index = ["a","b","c"], columns = ["d1","d2","d3"]), counts = True)
		self.mat = gt.spectral.adjacency(model.g, model.g.edge_properties["count"])[len(model.documents):,:len(model.documents)]
		print(self.mat)
		self.assertEqual(self.mat.toarray()[0][0],1.0)

	def test_approx_nocounts(self):
                import pandas as pd
                model = self.model
                self.model.make_graph_from_BoW_df(pd.DataFrame(data=[[1.1,2.2,3.3],[0.5,0.2,0.1],[0.03,0.01,1.005]], index = ["a","b","c"], columns = ["d1","d2","d3"]), counts = True)
                self.mat = gt.spectral.adjacency(model.g, model.g.edge_properties["count"])[len(model.documents):,:len(model.documents)]
                print(self.mat)
                self.assertEqual(self.mat.toarray()[0][0],1.0)

if __name__=="__main__":
	unittest.main()
