import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mat
import json
# fr = open("Performance.csv", "r")
# ls = []
# for line in fr:
#     line = line.replace("\n", "")
#     ls.append(line.split(','))
# fr.close()
# fw = open("Performance.json", "w")
# for i in range(1, len(ls)):
#     ls[i] = dict(zip(ls[0], ls[i]))
# json.dump(ls[1:], fw, sort_keys=True, indent=4, ensure_ascii=False)
# fw.close()
node = np.loadtxt('node.csv', delimiter=',')
edge = np.loadtxt('edge.csv', delimiter=',')
label = node[:, 0].astype(np.int)
dis = node[:, 1:3]
label1, label2, label3 = label[:3], label[3:9], label[9:]
dis1, dis2, dis3 = dis[:3], dis[3:9], dis[9:]
G, G1, G2, G3 = nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()
# G.add_nodes_from(node)
G1.add_nodes_from(label1)
G2.add_nodes_from(label2)
G3.add_nodes_from(label3)
for col in edge:
    G.add_edges_from([(col[0], col[1])])
pos, pos1, pos2, pos3 = {}, {}, {}, {}
for i, (row1, row2) in enumerate(dis):
    pos.update({i + 1: [row1, row2]})
for i, (row1, row2) in enumerate(dis1):
    pos1.update({i + 1: [row1, row2]})
for i, (row1, row2) in enumerate(dis2):
    pos2.update({i + 4: [row1, row2]})
for i, (row1, row2) in enumerate(dis3):
    pos3.update({i + 10: [row1, row2]})
nx.draw_networkx(G, pos=pos, with_labels=False, node_color='r', node_size=1, width=0.2)
nx.draw_networkx(G1, pos=pos1, with_labels=True, nodelist=label1, node_color='c', node_size=200, font_size=10)
nx.draw_networkx(G2, pos=pos2, with_labels=True, nodelist=label2, node_color='b', node_size=200, font_color='w',
                 font_size=10)
nx.draw_networkx(G3, pos=pos3, with_labels=False, nodelist=label3, node_color='r', node_size=1)
plt.savefig("net.png")
plt.show()

# labels = '4G', '5G', '6G'
# sizes = [60, 30, 10]
# plt.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%d%%')
# plt.title('SupportedType')
# plt.savefig('SupportedType.png')
# plt.show()
#
# labels = "手机", "Pad", "电脑", "传感器"
# sizes = [50, 20, 25, 5]
# matplotlib.rcParams['font.family'] = 'SimHei'
# plt.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%d%%',)
# plt.title('TerminalType(cpe)')
# plt.savefig('TerminalType(cpe).png')
# plt.show()
# sizes = [80, 10, 5, 5]
# plt.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%d%%',)
# plt.title('TerminalType(cpn)')
# plt.savefig('TerminalType(cpn).png')
# plt.show()

# labels = "华为", "小米"
# sizes = [50, 50]
# mat.rcParams['font.family'] = 'SimHei'
# plt.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
# plt.title('TerminalBrand(传感器)')
# plt.savefig('TerminalBrand(传感器).png')
# plt.show()

# data = np.loadtxt('data.csv',  dtype=int, delimiter=',')
# plt.hist(data[:, 1], bins=30, edgecolor='black', histtype='bar', facecolor='b')
# plt.title('UpOctDL')
# plt.xlabel('UpOctDL(KB)')
# plt.ylabel('Frequency')
# plt.savefig('UpOctDL.png')
# plt.show()
