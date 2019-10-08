# Graph_Convlstm2predict
a method to predict population in metro train station of Hangzhou

this repositor is used to predict population in metro train station of Hangzhou( https://tianchi.aliyun.com/competition/entrance/231712/introduction?spm=5176.12281915.0.0.7db04226JyNI8C ).

here we integrate graph convolution net(https://github.com/mdeff/cnn_graph) and Convlstm(http://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf).

our net is constrcuted and trained based on pytorch.

graph.py is a util that include operation of graph convolution.

convlstm1.py is a util that include construct of Convlstm, we improve its convolution operation into graph-convolution operation.

dataloader.py is to load data.

train.py is to construct and train our net.

predict.py is to get the results, but it is too bad to read.

del_datas0.csv is data that include the number of arrivals at each station along time.

del_datas1.csv is data that include the number of departures at each station along time.

Metro_roadMap.csv is the metro train station road map expressed as adjacency matrix.
