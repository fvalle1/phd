from graph_tool.all import *
import graph_tool as gt
import matplotlib

g = Graph()

g.add_vertex()
g.add_vertex()
g.add_vertex()
g.add_vertex()
g.add_vertex()
g.add_vertex()
g.add_vertex()
g.add_vertex()
g.add_vertex()
g.add_vertex()
g.add_vertex()
v_kind = g.new_vertex_property("int")
v_kind[0]=0
v_kind[1]=0
v_kind[2]=0
v_kind[3]=0
v_kind[4]=0
v_kind[5]=0
v_kind[6]=1
v_kind[7]=1
v_kind[8]=1
v_kind[9]=1
v_kind[10]=1
g.list_properties()
g.add_edge(0,6)
g.add_edge(0,7)
g.add_edge(0,8)
g.add_edge(0,9)
g.add_edge(0,10)
g.add_edge(1,6)
g.add_edge(1,7)
g.add_edge(2,6)
g.add_edge(2,7)
g.add_edge(3,8)
g.add_edge(3,9)
g.add_edge(3,10)
g.add_edge(4,10)
g.add_edge(4,9)
g.add_edge(4,8)
g.add_edge(5,6)
g.add_edge(5,7)
g.add_edge(5,8)
g.add_edge(5,9)
g.add_edge(5,10)

v_color = g.new_vertex_property("vector<double>")
for vertex in g.vertices():
    v_color[vertex]=[0.640625, 0, 0, 0.9] if v_kind[vertex]==0 else [0,0,0.640625, 0.9]


state = gt.inference.minimize_nested_blockmodel_dl(g, deg_corr=True)
gt.draw.draw_hierarchy(state, output="draw.pdf", layout="bipartite", output_size=(1000, 1000), vertex_color=[0.1, 0.1, 0.1, 0.9],
          vertex_fill_color=v_color, vertex_size=50, edge_marker_size=0, edge_pen_width=10,
          vcmap=matplotlib.cm.gist_heat_r, deg_size=False, hvertex_size=0)
