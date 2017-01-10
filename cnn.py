from __future__ import division
import os
import gzip
import cPickle
import numpy as np



#Convolve
convolve = lambda X,W: np.sum(np.multiply(X,W))

#Maxpool
maxpool = lambda X: X.max()

#Relu
relu = lambda X: X if X > 0 else 0

#dRelu
drelu = lambda X: 1 if X>0 else 0

#Sigmoud
sigmoid = lambda X: 1/(1+np.exp(-X))

#dSigmoid
dsigmoid = lambda X: sigmoid(X)*(1-sigmoid(X))

#Compute activation dim
get_ouput_dim = lambda n,f,p,s: ((n-f+2*p)/s) + 1

#Reshape Train
reshape_train = lambda x: x.reshape(28,28,1)

#Convert Response to vectors
response_vec = lambda sample : np.array([1 if i == sample else 0 for i in range(10)]).reshape(10,1)


def load_data():
	f_path = 'datasets/mnist.pkl.gz'
	f = gzip.open(f_path,'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	f.close()
	return(training_data, validation_data, test_data)


#Maxpool_Indx
def maxpool_indx(X, ref_x, ref_y, stride): 
	i,j = np.unravel_index(X.argmax(), X.shape)
	max_i = stride*ref_x + i 
	max_j = stride*ref_y +j
	return max_i, max_j


#Transform Input Layer
def transform_X(X, conv_params):
	X_transf = np.zeros((conv_params['convlayer_dim']*conv_params['convlayer_dim'], conv_params['filter_size']*conv_params['filter_size'], X.shape[2]))
	for map_indx in range(X.shape[2]):
		row_indx = 0
		for i in range(0,X.shape[0]-conv_params['filter_size']+1,  conv_params['stride']): 
			for j in range(0,X.shape[0]-conv_params['filter_size']+1,  conv_params['stride']):
				row_indx += 1
				X_transf[row_indx-1,:,map_indx] = X[i:i+conv_params['filter_size'], j:j+conv_params['filter_size'], map_indx].flatten()
	return X_transf


def conv_layer(X_transf, kernel_map):
	activation_map = np.zeros((X_transf.shape[0],1,len(kernel_map))) 
	#Select Filter
	for filter_indx in range(len(kernel_map)):
		#Get Current Filter
		filter = kernel_map[filter_indx]
		convolution_per_channel = np.zeros((X_transf.shape[0],1,X_transf.shape[2]))
		#Iterate over channels
		for channel_indx in range(X_transf.shape[2]):
			X_map = X_transf[:,:,channel_indx]
			K_map = filter[:,:, channel_indx]
			#Compute Product
			convolution_per_channel[:,:,channel_indx] = np.dot(X_map, K_map)
		#Sum across channels
		convolution_across_channels = np.sum(convolution_per_channel, axis=2)
		activation_map[:,:,filter_indx] = convolution_across_channels
	return activation_map


#Compute a maxpooling layer
def maxpooling_layer(inpt_volume, maxpool_params):
	
	maxpool_map = np.zeros((maxpool_params['maxpool_dim'], maxpool_params['maxpool_dim'], inpt_volume.shape[2]))
	maxpool_indices = []

	for f_indx in range(inpt_volume.shape[2]):
		filter_map = inpt_volume[:,:, f_indx]

		row_indx = 0
		for i in range(0, filter_map.shape[0]-maxpool_params['pooling_size']+1, maxpool_params['stride']):
			row_indx += 1
			col_indx = 0
			for j in range(0, filter_map.shape[0]-maxpool_params['pooling_size']+1, maxpool_params['stride']):
				col_indx += 1
				maxpool_map[row_indx-1, col_indx-1, f_indx] = maxpool(filter_map[i:i+maxpool_params['pooling_size'], j:j+maxpool_params['pooling_size']])
				max_i, max_j = maxpool_indx(filter_map[i:i+maxpool_params['pooling_size'], j:j+maxpool_params['pooling_size']], row_indx-1, col_indx-1, maxpool_params['stride'])
				maxpool_indices.append((max_i, max_j))

	return maxpool_map, maxpool_indices			

#Compute an activation layer
def non_linearity(activation_function, inpt):
	activated_input = map(activation_function, inpt.flatten())
	return np.array(activated_input).reshape(inpt.shape)


#Generate Filters
def generate_filters(filter_size, no_of_filters, filter_depth,):
	filter_map = [np.random.randn(filter_size*filter_size, 1, filter_depth) for i in range(no_of_filters)]
	return filter_map

#Reshape before MaxPooling
def reshape(inpt, convlayer_dim):

	map_depth = inpt.shape[2]
	reshaped_inpt = np.zeros((convlayer_dim, convlayer_dim, map_depth))

	for map_indx in range(map_depth):
		inpt_map = inpt[:,:,map_indx] 
		inpt_rmap = inpt_map.reshape(convlayer_dim,convlayer_dim)
		reshaped_inpt[:,:,map_indx] = inpt_rmap
	
	return reshaped_inpt

#Get Input Parameters for the CNN
def get_params(X):
	#X = np.random.randint(0,255,size=(7,7,3))
	#conv_params = {'filter_size':14 , 'stride':7 , 'zero_padding':0, 'no_filters': 1}
	conv_params = {'filter_size':10 , 'stride':2 , 'zero_padding':0, 'no_filters': 1}
	conv_params['convlayer_dim'] = get_ouput_dim(X.shape[0], conv_params['filter_size'],conv_params['zero_padding'], conv_params['stride'])
	maxpool_params = {'pooling_size':2 , 'stride':1, 'zero_padding':0 }
	maxpool_params['maxpool_dim'] = get_ouput_dim(conv_params['convlayer_dim'], maxpool_params['pooling_size'],maxpool_params['zero_padding'], maxpool_params['stride'])
	
	#Generate Parameters
	kernel_map = generate_filters(conv_params.get('filter_size'), conv_params.get('no_filters'), X.shape[2]) 

	#Generate Fully Connected Params
	fc_params = {'hidden_layer': 30, 'output_layer': 10, 'loss': 'binary_cross_entropy', 'h_inpt': maxpool_params.get('maxpool_dim')*maxpool_params.get('maxpool_dim')*conv_params.get("no_filters")}
	
	W1 = np.random.randn(fc_params.get('h_inpt'), fc_params.get('hidden_layer'))
	W2 = np.random.randn(fc_params.get('hidden_layer'), fc_params.get('output_layer')) 

	fc_weights = [W1, W2]

	return conv_params, maxpool_params, fc_params, kernel_map, fc_weights 


#dmaxpool
def dmaxpool(X, indices):
	Y = np.zeros(X.shape)
	for i,j in indices: Y[i,j] = 1
	return Y.reshape(Y.shape[0]*Y.shape[0],1, Y.shape[2])


#Gradient from hidden through maxpool to convolutional layer	
def gradient_to_convolutions(k, ind, conv_dim):
	conv_gradient = np.zeros((conv_dim,conv_dim,1))
	for indx in range(conv_gradient.shape[2]):
		for itm in range(len(ind)):
			i,j = ind[itm]
			val = k[itm]
			conv_gradient[i,j,indx] = val
	return conv_gradient



#LOAD DATA
train, valid, test = load_data()
mod_train = map(reshape_train,train[0])
mod_response = map(response_vec, train[1])
mod_test = map(reshape_train, test[0])
test_resp = test[1]
X_smpl = mod_train[0]
# Y = response_vec(train[1][0])
# print X.shape
#input & params
conv_params, maxpool_params, fc_params, kernel_map, fc_weights =  get_params(X_smpl)
alpha = 0.001

for epoch in range(15):
	print '\n>EPOCH: ',epoch
	for smpl_indx in range(len(mod_train)):
		#print '\n>Iter: ', smpl_indx
		X = mod_train[smpl_indx]
		Y = mod_response[smpl_indx]
		#print X.shape
		#print Y.shape
		##FORWARD-PROPAGATION

		#Transform_Input
		X_transf = transform_X(X, conv_params)
		#Convolution
		zc = conv_layer(X_transf, kernel_map)
		#Activation
		ac = non_linearity(relu, zc)
		#dActivation_gradient
		gradient_ac = non_linearity(drelu, ac)
		#Reshape For Max Pooling
		ac_rmap = reshape(ac, conv_params.get('convlayer_dim'))
		#MaxPool
		ac_maxpooled, maxpool_indices = maxpooling_layer(ac_rmap,maxpool_params)
		#gradient_ac_maxpooled -> Gate the gradient to dowonsampled values, block the excluded values
		gradient_maxpooling = dmaxpool(ac_rmap, maxpool_indices)
		#MaxPool-Flatten
		vectorized_maxpool = ac_maxpooled.flatten().reshape(len(ac_maxpooled.flatten()),1)

		#FullyConnected_ForwardProp
		z1 = np.dot(fc_weights[0].T, vectorized_maxpool)
		a1 = non_linearity(sigmoid, z1)
		z2 = np.dot(fc_weights[1].T, a1)
		a2 = non_linearity(sigmoid, z2)
		#PREDICTION: 
		#print '>>>', np.argmax(a2), np.argmax(Y)



		##BACKWARD-PROPAGATION
		dWc = []
		for filter in kernel_map: dWc.append(np.zeros(filter.shape))

		del_2 = a2 - Y
		del_1 = np.multiply(np.dot(fc_weights[1], del_2), non_linearity(dsigmoid, z1))
		err_m = np.dot(fc_weights[0], del_1)
		err_c = gradient_to_convolutions(err_m.flatten(), maxpool_indices, conv_params.get('convlayer_dim'))
		del_c = np.multiply(err_c.reshape(gradient_ac.shape), gradient_ac)
		dw2 =  np.dot(a1, del_2.T)
		dw1 =  np.dot(vectorized_maxpool, del_1.T)

		#Gradient for Filters WCs
		for channel_indx in range(X_transf.shape[2]): 
			for filter_indx in range(len(dWc)):
				dWc[filter_indx][:,:, channel_indx] = np.dot(X_transf[:,:,channel_indx].T, del_c[:,:,channel_indx])


		##Weight Updates
		fc_weights[1] -= alpha*dw2
		fc_weights[0] -= alpha*dw1

		for filter_indx in range(len(kernel_map)): 
			for channel_indx in range(kernel_map[filter_indx].shape[2]):
				kernel_map[filter_indx][:,:,channel_indx] -= alpha*dWc[filter_indx][:,:,channel_indx]


#TEST
acc = 0
for sample_indx in range(len(mod_test)):
	sample = mod_test[sample_indx]
	resp = test_resp[sample_indx]	
	#Transform_Input
	X_transf = transform_X(sample, conv_params)
	#Convolution
	zc = conv_layer(X_transf, kernel_map)
	#Activation
	ac = non_linearity(relu, zc)
	#Reshape For Max Pooling
	ac_rmap = reshape(ac, conv_params.get('convlayer_dim'))
	#MaxPool
	ac_maxpooled, maxpool_indices = maxpooling_layer(ac_rmap,maxpool_params)
	#MaxPool-Flatten
	vectorized_maxpool = ac_maxpooled.flatten().reshape(len(ac_maxpooled.flatten()),1)

	#FullyConnected_ForwardProp
	z1 = np.dot(fc_weights[0].T, vectorized_maxpool)
	a1 = non_linearity(sigmoid, z1)
	z2 = np.dot(fc_weights[1].T, a1)
	a2 = non_linearity(sigmoid, z2)
	if np.argmax(a2) == resp: acc+=1


print '\n\n\nRESULTS: ', acc, len(mod_test), acc*100/len(mod_test)


#epoch1: 60
#epoch5: 83.1