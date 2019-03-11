from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell

from lib import utils


class DCGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, args, num_units, adj_mx, squeeze_and_excitation, se_activate, excitation_rate, r, diffusion_with_graph_kernel, graph_kernel_mode, 
                 cell_forward_mode, max_diffusion_step, num_nodes, num_proj=None,
                 activation=tf.nn.tanh, reuse=None, filter_type="laplacian", use_gc_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """
        super(DCGRUCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_nodes = num_nodes
        #self._num_edges = np.sum(adj_mx!=0)
        self.mask_mx = adj_mx!=0
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self.squeeze_and_excitation = squeeze_and_excitation
        self.se_activate = se_activate
        self.excitation_rate = excitation_rate
        self.r = r
        self.cell_forward_mode = cell_forward_mode
        self.diffusion_channel_num = args.diffusion_channel_num
        self.diffusion_with_graph_kernel = diffusion_with_graph_kernel
        self.graph_kernel_mode = graph_kernel_mode
        supports = []
        print('adj_mx: ', adj_mx.shape)
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))
        self._kernel_inds = self.kernel_ind(supports)  
        self.mask_mx_ind = self.mask_ind(supports)
       
    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        with tf.variable_scope(scope or "dcgru_cell"):
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                output_size = 2 * self._num_units
                # We start with bias of 1.0 to not reset and not update.
                if self.cell_forward_mode=='gconv':
                    fn = self._gconv
                elif self.cell_forward_mode=='fc':
                    fn = self._fc
                value = tf.nn.sigmoid(fn(inputs, state, output_size, bias_start=1.0))
                value = tf.reshape(value, (-1, self._num_nodes, output_size))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
                r = tf.reshape(r, (-1, self._num_nodes * self._num_units))
                u = tf.reshape(u, (-1, self._num_nodes * self._num_units))
            with tf.variable_scope("candidate"):
                c = fn(inputs, r * state, self._num_units)
                if self._activation is not None:
                    c = self._activation(c)
                    c = tf.reshape(c, (-1, self._num_nodes* self._num_units))    

            output = new_state = u * state + (1 - u) * c
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_state, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = tf.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=-1)
        input_size = inputs_and_state.get_shape()[-1].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
        value = tf.nn.sigmoid(tf.matmul(inputs_and_state, weights))
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)
        x0_backups = x0
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            if self._max_diffusion_step == 0:
                num_matrices = 1
            elif not self.diffusion_with_graph_kernel:
                for support in self._supports:
                    x0 = x0_backups
                    x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

                num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
            else:
                for i, support in enumerate(self._supports): 
                    x1 = [x0]
                    for j in range(self._max_diffusion_step):
                        x1 = self.graph_kernel_conv(j, self.diffusion_channel_num[j], x1, support, i, self.graph_kernel_mode)
                        for x2 in x1:
                            x = self._concat(x, x2)   
                num_matrices = len(self._supports) * sum(self.diffusion_channel_num) + 1  
             
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            if self.squeeze_and_excitation:
                x = tf.reshape(x, shape=[batch_size*self._num_nodes*input_size, num_matrices])
                theta_x = self.se_fc(x, num_matrices, '1', dtype, bias_start)
                x = tf.multiply(x, theta_x)
                x = tf.reshape(x, shape=[batch_size*self._num_nodes*input_size, num_matrices])                 
            
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
            weights = tf.get_variable(
                   'weights', [input_size * num_matrices, output_size], dtype=dtype,
                    initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
           
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])
        
    def se_fc(self, x, output_num, i, dtype, bias_start):
        input_num = x.shape[1]
        se_weights = tf.get_variable('se_weights'+i, [input_num, output_num], dtype=dtype,
                               initializer=tf.contrib.layers.xavier_initializer())
        se_biases = tf.get_variable("se_biases1"+i, [output_num], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype)) 
        x = tf.matmul(x, se_weights)+se_biases   
        return x  

    def channel_wise_attention(self, inputs, attention_mode, r, bias_start=0.0):
        batch_size = inputs.shape[0] 
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        channel_size = inputs.shape[2]
        hidden_feature_size = channel_size//r
        dtype = inputs.dtype
        
        if attention_mode=='global':
            with tf.variable_scope('channel_wise_'+attention_mode+'_attention'):
                squeeze_inputs = tf.reduce_sum(inputs, 1)
                first_outputs = tf.nn.relu(self.se_fc(squeeze_inputs, hidden_feature_size, '1', dtype, bias_start))
                excitation = tf.nn.sigmoid(self.se_fc(first_outputs, channel_size, '2', dtype, bias_start))
                excitation = tf.expand_dims(excitation, 1)
                return tf.reshape(tf.multiply(inputs, excitation), shape=[batch_size, self._num_nodes*int(channel_size)])
        elif attention_mode=='local':
            with tf.variable_scope('channel_wise_'+attention_mode+'_attention'):
                local_inputs = tf.reshape(inputs, shape=[batch_size*self._num_nodes, channel_size])
                first_outputs = tf.nn.relu(self.se_fc(local_inputs, hidden_feature_size, '1', dtype, bias_start))     
                local_excitation = tf.nn.sigmoid(self.se_fc(first_outputs, channel_size, '2', dtype, bias_start))
                local_excitation = tf.reshape(local_excitation, shape=[batch_size, self._num_nodes, channel_size])
                return tf.reshape(tf.multiply(inputs, local_excitation), shape=[batch_size, self._num_nodes*int(channel_size)]) 

    def diffusion(self, inputs):
        batch_size = inputs.shape[0] 
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))   
        channel_size = inputs.shape[2]
        inputs0 = tf.transpose(inputs, perm=[1, 2, 0])  
        inputs0 = tf.reshape(inputs0, shape=[self._num_nodes, channel_size * batch_size])
        inputs = tf.expand_dims(inputs0, axis=0)
        inputs0_backups = inputs0
        if self._max_diffusion_step == 0:
            num_matrices = 1
        elif not self.diffusion_with_graph_kernel:
            for support in self._supports:
                inputs0 = inputs0_backups
                inputs1 = tf.sparse_tensor_dense_matmul(support, inputs0)
                inputs = self._concat(inputs, inputs1)
                for k in range(2, self._max_diffusion_step + 1):
                    inputs2 = 2 * tf.sparse_tensor_dense_matmul(support, inputs1) - inputs0
                    inputs = self._concat(inputs, inputs2)
                    inputs1, inputs0 = inputs2, inputs1
            num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for inputs itself.
        else:
             for i, support in enumerate(self._supports): 
                 inputs1 = [inputs0]
                 for j in range(self._max_diffusion_step):
                        inputs1 = self.graph_kernel_conv(j, self.diffusion_channel_num[j], inputs1, support, i, self.graph_kernel_mode)
                        for inputs2 in inputs1:
                            inputs = self._concat(inputs, inputs2)    
             num_matrices = len(self._supports) * sum(self.diffusion_channel_num) + 1        
                    
        inputs = tf.reshape(inputs, shape=[num_matrices, self._num_nodes, channel_size, batch_size])
        inputs = tf.transpose(inputs, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order) 
        inputs = tf.reshape(inputs, shape=[batch_size, num_matrices*self._num_nodes*int(channel_size)])      
        return inputs   

    def kernel_ind(self, matrix):
        inds = []
        for mat in matrix:
            A = np.array(mat.todense())
            ind = np.flip(np.argsort(np.argsort(A, axis=1), 1), 1)
            inds.append(ind)
        return inds    

    def graph_kernel(self, weights, ind):
        return tf.gather(weights, ind)
    
    def graph_kernel_conv(self, scope, channel_num, inputs, matrix, matrix_serial_num, graph_kernel_mode):
        with tf.variable_scope(f'graph_kernel_conv_matrix_{matrix_serial_num+1:1d}_layer_{scope+1:1d}'):
            if graph_kernel_mode=='local':
                #weights = self.ones([channel_num*len(inputs), self._num_nodes], 'graph_kernel')
                weights = tf.get_variable('graph_kernel', [channel_num*len(inputs), self._num_nodes], dtype=inputs[0].dtype, initializer=tf.contrib.layers.xavier_initializer())  
                outputs = []
                for i in range(channel_num):
                    for k, inp in enumerate(inputs):
                        w = weights[i*len(inputs)+k, :] 
                        w = self.graph_kernel(w, self._kernel_inds[matrix_serial_num])
                        if k==0:
                            m = matrix*w
                            output = tf.sparse_tensor_dense_matmul(m, inp)
                        else:
                            m = matrix*w
                            output += tf.sparse_tensor_dense_matmul(m, inp)    
                    outputs.append(output)
            elif graph_kernel_mode=='global':
                #weights = self.ones([channel_num*len(inputs), self._num_edges], 'graph_kernel')
                weights = tf.get_variable('graph_kernel', [channel_num*len(inputs), self._num_edges], dtype=inputs[0].dtype, initializer=tf.contrib.layers.xavier_initializer()) 
                outputs = []
                for i in range(channel_num):
                    for k, inp in enumerate(inputs):
                        w = weights[i*len(inputs)+k, :] 
                        w = self.attention_kernel(w, self.mask_mx_ind[matrix_serial_num])
                        if k==0:
                            m = matrix*w
                            output = tf.sparse_tensor_dense_matmul(m, inp)
                        else:
                            m = matrix*w
                            output += tf.sparse_tensor_dense_matmul(m, inp)    
                    outputs.append(output)      
        return outputs 

    def mask_ind(self, matrix):
        inds = []
        for mat in matrix:
            A = np.array(mat.todense())
            ind = A!=0 
            inds.append(ind)
        self._num_edges = np.sum(A!=0)    
        return inds  
    
    def attention_kernel(self, weights, mask):
        ind = np.zeros((self._num_nodes, self._num_nodes), dtype=int)
        w0 = tf.zeros(1)
        w = tf.concat((w0, weights), 0)
        ind[mask] = np.array(range(self._num_edges)) + 1
        return tf.gather(w, ind)
    '''   
    def arr2sparse(arr_tensor):
    	   arr_tensor = tf.constant(np.array(arr), dtype=tf.float32)
        arr_idx = tf.where(tf.not_equal(arr_tensor, 0))
        arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(arr_tensor, arr_idx), arr_tensor.get_shape())
        return arr_sparse  
    '''      
    
    def ones(self, shape, name=None):
        """All ones."""
        initial = tf.ones(shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)
             