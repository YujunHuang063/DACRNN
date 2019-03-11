from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import time
import yaml

from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import masked_mae_loss, masked_rmse_loss, masked_mape

from model.dcrnn_model import DCRNNModel


class DCRNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, args, adj_mx, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, kwargs['name']+'_info.log', level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)
        self._logger.info(kwargs)
 
        # Data preparation
        if self._data_kwargs.get('data_type')=='npz':
            self._data = utils.load_dataset(**self._data_kwargs)
        elif self._data_kwargs.get('data_type')=='csv':
            self._data = utils.load_dataset_from_csv(**self._data_kwargs)    
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Build models.
        scaler = self._data['scaler']
        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                self._train_model = DCRNNModel(args=args, is_training=True, scaler=scaler,
                                               batch_size=self._data_kwargs['batch_size'],
                                               adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                self._test_model = DCRNNModel(args=args, is_training=False, scaler=scaler,
                                              batch_size=self._data_kwargs['test_batch_size'],
                                              adj_mx=adj_mx, **self._model_kwargs)

        # Learning rate.
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.01),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Configure optimizer
        optimizer_name = self._train_kwargs.get('optimizer', 'adam').lower()
        epsilon = float(self._train_kwargs.get('epsilon', 1e-3))
        optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr, )
        elif optimizer_name == 'amsgrad':
            optimizer = AMSGrad(self._lr, epsilon=epsilon)

        # Calculate loss
        output_dim = self._model_kwargs.get('output_dim')
        preds = self._train_model.outputs
        labels = self._train_model.labels[..., :output_dim]

        null_val = 0.
        self._loss_fn = masked_mae_loss(scaler, null_val)
        self._mape_fn = masked_mape(scaler, null_val)
        self._rmse_fn = masked_rmse_loss(scaler, null_val)
        self._train_loss = self._loss_fn(preds=preds, labels=labels)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, tvars)
        max_grad_norm = kwargs['train'].get('max_grad_norm', 1.)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        global_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')

        max_to_keep = self._train_kwargs.get('max_to_keep', 100)
        self._epoch = 0
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s_' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))+kwargs['name']+'/'
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def run_epoch_generator(self, sess, mode, model, data_generator, scale, return_output=False, training=False, writer=None):
        losses = []
        maes = []
        outputs = []
        output_dim = self._model_kwargs.get('output_dim')
        preds = model.outputs
        labels = model.labels[..., :output_dim]
        loss = self._loss_fn(preds=preds, labels=labels)
        mae_horizon={}
        mape_horizon={}
        rmse_horizon={}
        fetches = {
            'loss': loss,
            'mae': loss,
            'global_step': tf.train.get_or_create_global_step()
        }
        if mode=='test':
            for i in range(1, 13):
                fetches.update({
                    'mae_'+str(i): self._loss_fn(preds=preds[:, i-1, :, :], labels=labels[:, i-1, :, :])
                })
                mae_horizon[i] = []
                fetches.update({
                    'mape_'+str(i): self._mape_fn(preds=preds[:, i-1, :, :], labels=labels[:, i-1, :, :])
                })
                mape_horizon[i] = []
                fetches.update({
                    'rmse_'+str(i): self._rmse_fn(preds=preds[:, i-1, :, :], labels=labels[:, i-1, :, :])
                })
                rmse_horizon[i] = []
        
        if training:
            fetches.update({
                'train_op': self._train_op
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs
            })
        labels = []
        for i, (x, y) in enumerate(data_generator):
            #print('train_y_data_max: ', np.max(y))
            feed_dict = {
                model.inputs: np.clip(x, -10, 10),
                model.labels: np.clip(y, -10, 10),
            }

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            if mode=='test':
                for j in range(1, 13):
                    mae_horizon[j].append(vals['mae_'+str(j)])
                    mape_horizon[j].append(vals['mape_'+str(j)])
                    rmse_horizon[j].append(vals['rmse_'+str(j)])
            maes.append(vals['mae'])
            if i % 30==0:
                print(i)
                print(vals['loss'])
                print(vals['mae'])
                print('pred: ', vals['outputs'][0,0:1,5:6,0])
                print(vals['outputs'].shape)
                
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])
            labels.append(y)
                
        _outputs = np.concatenate(outputs, 0)
        truth = np.concatenate(labels, 0)
        if mode=='test':
            for j in range(1, 13):
                self._logger.info('horizon: %d mae: %f mape: %f rmse: %f'%(j, np.mean(mae_horizon[j]), np.mean(mape_horizon[j]), np.mean(rmse_horizon[j])))        

        results = {
            'loss': np.mean(losses),
            'mae': np.mean(maes)
        }
        if return_output:
            results['outputs'] = outputs
        return results

    def get_lr(self, sess):
        return np.asscalar(sess.run(self._lr))

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def _train(self, sess, base_lr, epoch, steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10, save_epoch_interval=5, **train_kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        max_to_keep = train_kwargs.get('max_to_keep', 100)
        model_metaname = train_kwargs.get('model_metaname')
        if model_metaname is not None:
            pass
            #saver = tf.train.import_meta_graph(os.path.join('data/model/dcrnn_DR_2_h_12_64-64_lr_0.005_bs_32_0131205604_test', model_metaname))
        else:    
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
            
        model_filename = train_kwargs.get('model_filename')

        if model_filename is not None:
            saver.restore(sess, os.path.join(self._kwargs['base_dir'], model_filename))
            self._epoch = epoch + 1
        else:
            sess.run(tf.global_variables_initializer())
        self._logger.info('Start training ...')

        while self._epoch <= epochs:
            
            # Learning rate schedule.
            new_lr = max(min_learning_rate, base_lr * (lr_decay_ratio ** np.sum(self._epoch >= np.array(steps))))
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            train_results = self.run_epoch_generator(sess, 'train', self._train_model,
                                                     self._data['train_loader'].get_iterator(),
                                                     None,
                                                     training=True,
                                                     return_output=True,
                                                     writer=self._writer)
            train_loss, train_mae = train_results['loss'], train_results['mae']
            if train_loss > 1e5:
                self._logger.warning('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            
            # Compute validation error.
            print('--------------------------------------------------------------------------------')
            val_results = self.run_epoch_generator(sess, 'val', self._test_model,
                                                   self._data['val_loader'].get_iterator(),
                                                   None,
                                                   return_output=True,
                                                   training=False)
            val_loss, val_mae = np.asscalar(val_results['loss']), np.asscalar(val_results['mae'])
            y_preds = val_results['outputs']
            scaler = self._data['scaler']
            y_preds = np.concatenate(y_preds, axis=0)
            for horizon_i in range(self._data['y_val'].shape[1]):
                y_truth = scaler.inverse_transform(self._data['y_val'][:, horizon_i, :, 0])
                print('truth',y_truth[0,:5])

                y_pred = scaler.inverse_transform(y_preds[0:5, horizon_i, :, 0])
                print('pred',y_pred[0,:5])            
            
            utils.add_simple_summary(self._writer,
                                     ['loss/train_loss', 'metric/train_mae', 'loss/val_loss', 'metric/val_mae'],
                                     [train_loss, train_mae, val_loss, val_mae], global_step=global_step)
            end_time = time.time()
            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} lr:{:.6f} {:.1f}s'.format(
                self._epoch, epochs, global_step, train_mae, val_mae, new_lr, (end_time - start_time))
            self._logger.info(message)
            if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
                self.evaluate(sess)
            if val_loss<min_val_loss:
                 self._logger.info(('Val loss decrease from %.4f to %.4f') % (min_val_loss, val_loss))
                 min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    self._logger.warning('Early stopping at epoch: %d' % self._epoch)
                    break             
            if self._epoch % save_epoch_interval == 0:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(sess, val_loss)
                    self._logger.info('min Val loss  %.4f ,Val loss %.4f, saving to %s' % (min_val_loss, val_loss, model_filename))     #model_filename


            history.append(val_mae)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
        return np.min(history)

    def evaluate(self, sess, **kwargs):
        global_step = sess.run(tf.train.get_or_create_global_step())
        test_results = self.run_epoch_generator(sess, 'test', self._test_model,
                                                self._data['test_loader'].get_iterator(),
                                                self._data['scaler'], 
                                                return_output=True,
                                                training=False)
          
        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds = test_results['loss'], test_results['outputs']
        self._logger.info('test_mae: %f', (np.asscalar(test_loss)))
        utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)
        return

    def load(self, sess, model_filename):
        """
        Restore from saved model.
        :param sess:
        :param model_filename:
        :return:
        """
        self._saver.restore(sess, model_filename)

    def save(self, sess, val_loss):
        config = dict(self._kwargs)
        global_step = np.asscalar(sess.run(tf.train.get_or_create_global_step()))
        prefix = os.path.join(self._log_dir, 'models-{:.4f}'.format(val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self._saver.save(sess, prefix, global_step=self._epoch,
                                                             write_meta_graph=True)
        config_filename = 'config_{}.yaml'.format(self._epoch)
        '''
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        '''    
        return config['train']['model_filename']
