# Ex3_OOp

### This README explains the 3rd assignment we got on OOP course in Ariel university.
### In this project we were asked to implement a Python weighted directed graph.

#### The assignment has three parts:
#### The first and the second part is to create a directed weighted graph in python 

there were 3 classes for this part:

•	nodeData represents the nodes in the graph, each node has some information and a geoLocation object.

•	DiGraph represents the graph itself, includes nodes, edges and mode account that count the changes in the graph. the main functions in this class are:

1.	Add_node - we need to give this method a key (and position- optional) we want to add to the graph if there is such key it will do nothing , add it otherwise.

2.	remove_node - we need to give this method a key and it will delete the node from the graph including all his edges.

3.	remove_edge - we need to give this method 2 keys and it will remove the specific edge between them if there is no edge like this it will do nothing.

4.	v_size - this method will return the number of the vertices in the graph.

5.	e_size - this method will return the number of the edges in the graph.

6.	get_MC - this method will return the number of the changes in the graph.

7.	get_all_v- this method will return a dictionary of all the nodes in the Graph, each node is represented using a pair (node_id, node_data).

8.  all_in/out_edges_of_node- this method will return a dictionary of all the nodes connected to (into)/from node_id , each node is represented using a pair (other_node_id, weight).


•	GraphAlgo is the class that made to use any given graph, the functions in this class are shortest way between two nodes, connected components and component, save, load, get graph and plot. the main functions in this class are:

1.  get_graph- this method will return the directed graph on which the algorithm works on.

2.  load_from_json- this method Loads a graph from a json file. returns True if the loading was successful, False o.w.

3.  save_to_json- This method Saves the graph in JSON format to a file. it will return True if the save was successful, False o.w.
 
4.	shortest_path - we need to give 2 keys of nodes and this method will reurns a List of nodes of the shortest path between them if no such path it will returns null.

5.	connected_component- this method Finds the Strongly Connected Component(SCC) that node id1 is a part of.it will return The list of nodes in the SCC. If the graph is None or id1 is not in the graph it will return an empty list [].

6.  connected_components- this method Finds all the Strongly Connected Component(SCC) in the graph. It will return the list all SCC.  If the graph is None the function will return an empty list [].

7.  plot_graph- this method Plots the graph. If the nodes have a position, the nodes will be placed there. Otherwise, they will be placed in a random but elegant manner.

for example:
![](https://user-images.githubusercontent.com/57614822/104652427-cfce1b00-56c1-11eb-8840-68cfb01c4035.jpeg)







#### The third part is to compare our graph vs NetworkX vs Java:
In this part we compared the different programs. 
here are the results:
![](https://user-images.githubusercontent.com/57614822/104653702-af06c500-56c3-11eb-89b0-c91143fa1b72.jpeg)

NetworkX is a Python library for studying graphs and networks. NetworkX is free software released under the BSD-new license.

We wrote tests to check the time differences
between our graph in python, NetworkX and Java. The function we were compared are shortest path, connected components and connected component (NetworkX does not have connected component).

more explanations on this wiki page

This assignment has written by Gil Zioni and Itamar Shpitzer, the course git is here.

