from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import yaml

classes=np.array(['Adipose Tissue', 'Blood', 'Brain', 'Breast', 'Lung', 'Pancreas',
       'Skin', 'Testis', 'Thyroid'])

y_test=np.genfromtxt("test.txt", dtype=int)


model_xml = "sequential_1.xml"
model_bin = "sequential_1.bin"

# Plugin initialization for specified device
plugin = IEPlugin(device="MYRIAD")

net = IENetwork(model=model_xml, weights=model_bin)
exec_net = plugin.load(network=net)

print("Predicting using MYRIAD")
for i,data in enumerate(np.genfromtxt('input.txt')):
	res = exec_net.infer(inputs={'dense_1_input': data})
	#print(res['dense_2/Softmax'])
	print("real: %s \t predicted: %s"%(classes[y_test[i]],classes[np.argmax(res['dense_1/Softmax'])]))

print(yaml.dump(exec_net.requests[0].get_perf_counts(), default_flow_style=False))
