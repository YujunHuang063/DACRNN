from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import legacy_seq2seq

from lib.metrics import masked_mae_loss
from model.dcrnn_cell import DCGRUCell


class DCRNNModel(object):
    def __init__(self, args, is_training, batch_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mae = None
        self._train_op = None

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        diffusion_with_graph_kernel = model_kwargs.get('diffusion_with_graph_kernel', False)
        graph_kernel_mode = model_kwargs.get('graph_kernel_mode', 'local')
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))
        pred_without_zero_input = model_kwargs.get('pred_without_zero_input')
        squeeze_and_excitation = model_kwargs.get('squeeze_and_excitation')
        se_activate = model_kwargs.get('se_activate')
        excitation_rate = model_kwargs.get('excitation_rate')
        r = model_kwargs.get('r')
        residuals = model_kwargs.get('residuals')
        cell_forward_mode = model_kwargs.get('cell_forward_mode')
        with_inputs_diffusion = model_kwargs.get('with_inputs_diffusion')
        with_inputs_channel_wise_attention = model_kwargs.get('with_inputs_channel_wise_attention')
        attention_mode = model_kwargs.get('attention_mode')
        
        aux_dim = input_dim - output_dim

        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')

        # GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * input_dim))
        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * output_dim))
        init_state = [tf.zeros(shape=(batch_size, num_nodes * rnn_units)), tf.zeros(shape=(batch_size, num_nodes * rnn_units))]
        cell = DCGRUCell(args, rnn_units, adj_mx, squeeze_and_excitation, se_activate, excitation_rate, r, diffusion_with_graph_kernel, graph_kernel_mode,
                         cell_forward_mode, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type, use_gc_for_ru=True)
        cell_with_projection = DCGRUCell(args, rnn_units, adj_mx, squeeze_and_excitation, se_activate, excitation_rate, r, diffusion_with_graph_kernel, graph_kernel_mode,
                                         cell_forward_mode,max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                         num_proj=output_dim, filter_type=filter_type, use_gc_for_ru=True)
        test_cells = [cell] * num_rnn_layers                                 
        #encoding_cells = [cell] * num_rnn_layers
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        #encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        test_cells = tf.contrib.rnn.MultiRNNCell(test_cells, state_is_tuple=True)
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)
        

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            inputs = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            labels = tf.unstack(
                tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1)
            if aux_dim > 0:
                aux_info = tf.unstack(self._labels[..., output_dim:], axis=1)
                aux_info.insert(0, None)
            if not pred_without_zero_input:
                labels.insert(0, GO_SYMBOL)

            def _loop_function(prev, i):
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if use_curriculum_learning:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: labels[i], lambda: prev)
                    else:
                        result = labels[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                if False and aux_dim > 0:
                    result = tf.reshape(result, (batch_size, num_nodes, output_dim))
                    result = tf.concat([result, aux_info[i]], axis=-1)
                    result = tf.reshape(result, (batch_size, num_nodes * input_dim))
                return result
                
            output, state = rnn_encoder(inputs, init_state, test_cells, cell, with_inputs_diffusion, with_inputs_channel_wise_attention, attention_mode, r)
            if pred_without_zero_input:
                first_output = get_first_output(num_nodes, rnn_units, output_dim, output)
                outputs, final_state = rnn_decoder(labels, state, decoding_cells, cell, with_inputs_diffusion, 
                                                              with_inputs_channel_wise_attention, attention_mode, r, 
                                                              loop_function=_loop_function, )                                            
                outputs.insert(0, first_output)                                              
            else:    
            #_, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
                outputs, final_state = rnn_decoder(labels, state, decoding_cells, cell, with_inputs_diffusion, 
                                                              with_inputs_channel_wise_attention, attention_mode, r, 
                                                              loop_function=_loop_function)

        # Project the output to output_dim.
        
        if  residuals and not pred_without_zero_input:
            first_base_outputs = tf.reshape(tf.reshape(inputs[-1], (batch_size, num_nodes, input_dim))[..., :output_dim], (batch_size,  num_nodes * output_dim))
            base_outputs = outputs[:-2]
            base_outputs.insert(0, first_base_outputs)
            base_outputs = tf.stack(base_outputs, axis=1)
            outputs = tf.stack(outputs[:-1], axis=1) 
            outputs += base_outputs
        else:    
            outputs = tf.stack(outputs[:-1], axis=1)    
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')
        self._merged = tf.summary.merge_all()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs

def rnn_encoder(inputs, init_state, cell, cell_unit, with_inputs_diffusion, with_inputs_channel_wise_attention, attention_mode, r):
    state = init_state
    with tf.variable_scope('test_encoder'):
        for i in range(len(inputs)):
            if i > 0:
                tf.get_variable_scope().reuse_variables()    
            if  with_inputs_diffusion:
                _inputs = cell_unit.diffusion(inputs[i]) 
            else:
                _inputs = inputs[i]         
            if with_inputs_channel_wise_attention:
                _inputs = cell_unit.channel_wise_attention(_inputs, attention_mode, r)            
            output, state = cell(_inputs, state)

    return output, state    

def rnn_decoder(decoder_inputs,
                initial_state,
                cell, cell_unit, with_inputs_diffusion, with_inputs_channel_wise_attention, attention_mode, r,
                loop_function=None,
                scope=None):

  with tf.variable_scope(scope or "test_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with tf.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        tf.get_variable_scope().reuse_variables()
      if  with_inputs_diffusion:
          inp = cell_unit.diffusion(inp) 
      if with_inputs_channel_wise_attention:
          inp = cell_unit.channel_wise_attention(inp, attention_mode, r)
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state

def get_first_output(num_nodes, rnn_units, output_dim, inputs):
    with tf.variable_scope("first_output"):
        w = tf.get_variable('w', shape=(rnn_units, output_dim))
        batch_size = inputs.get_shape()[0].value
        output = tf.reshape(inputs, shape=(-1, rnn_units))
        output = tf.reshape(tf.matmul(output, w), shape=(batch_size, num_nodes*output_dim))  
    return output    
    