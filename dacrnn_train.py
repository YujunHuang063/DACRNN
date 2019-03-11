from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml


from lib.utils import load_graph_data, load_graph_data_from_csv
from model.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    tf.reset_default_graph()
    with open(args.config_filename) as f:
        with tf.Graph().as_default() as g:
            supervisor_config = yaml.load(f)
            graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
            if supervisor_config['data']['data_type']=='npz':
                sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
            elif supervisor_config['data']['data_type']=='csv':
                adj_mx = load_graph_data_from_csv(supervisor_config['data'].get('dataset_dir'))    
            tf_config = tf.ConfigProto()
            if args.use_cpu_only:
                tf_config = tf.ConfigProto(device_count={'GPU': 0})
            tf_config.gpu_options.allow_growth = True
            #tf_config.gpu_options.per_process_gpu_memory_fraction = 1
            with tf.Session(config=tf_config) as sess:
                supervisor = DCRNNSupervisor(args=args, adj_mx=adj_mx, **supervisor_config)

                supervisor.train(sess=sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--diffusion_channel_num', default=[2, 1], type=list)
    args = parser.parse_args()
    main(args)
