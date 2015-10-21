import sys
import os
import argparse
import numpy as np
import scipy.io
import time
import two_stream_pb2 as two_stream
from google.protobuf.text_format import Merge

caffe_path = '/home/lxc/work/tools/caffe-action/python'
sys.path.insert(0, caffe_path)
import caffe


def prepare_data(video_name, nframes, params):

    def sample_frames(nframes, params):
        if params[0].sample_method == 'equal':
            interval = max(p.interval for p in params)
            length = max(p.length for p in params)
            frid = np.linspace(0, nframes-interval-length, params[0].fpv)
        elif params[0].sample_method == 'all':
            frid = np.arange(nframes)
        return frid

    def crop(frame, param):
        if param.crop_method == 'none':
            crop_data = frame
        elif param.crop_method == 'cc':
            csz = param.crop_size
            [h, w] = frame.shape[-2:]
            d = np.floor(np.subtract([h, w], csz)/2)
            # +---------+
            # | 1     3 |
            # |    5    |
            # | 2     4 |
            # +---------+
            crop_data = np.concatenate((frame[:, :, 0:csz[0], 0:csz[1]],
                                        frame[:, :, h-csz[0]:h, 0:csz[1]],
                                        frame[:, :, 0:csz[0], w-csz[1]:w],
                                        frame[:, :, h-csz[0]:h, w-csz[1]:w],
                                        frame[:, :, d[0]:d[0]+csz[0], d[1]:d[1]+csz[1]]), axis=0)
        elif param.crop_method == 'c':
            csz = param.crop_size
            [h, w] = frame.shape[-2:]
            d = np.floor(np.subtract([h, w], csz)/2)
            crop_data = frame[:, :, d[0]:d[0]+csz[0], d[1]:d[1]+csz[1]]
        return crop_data

    data = {}
    frid = sample_frames(nframes, params)
    for param in params:
        data_folder = '/'.join((param.data_path.encode('ascii', 'replace'), video_name))
        length = param.length
        fpv = param.fpv
        for x in xrange(len(frid)):
            start_fr = frid[x]
            for i in xrange(length):
                # read one frame
                if param.data_type == two_stream.DataParameter.FLOW:
                    img_x = caffe.io.load_image('%s/flow_x_%04d.jpg'%(data_folder, start_fr+i+1), color=False)
                    img_y = caffe.io.load_image('%s/flow_y_%04d.jpg'%(data_folder, start_fr+i+1), color=False)
                    frame = np.concatenate((img_x, img_y), axis=2)
                elif param.data_type == two_stream.DataParameter.RGB:
                    frame = caffe.io.load_image('%s/image_%04d.jpg' % (data_folder, start_fr+i+1))
                    frame = frame[:, :, (2, 1, 0)]
                frame *= 255.0  # caffe.io.load_image is in [0,1]
                # flip the frame
                if param.flip:
                    frame_flip = frame[:, ::-1, :]
                    if param.data_type == 'flow':
                        frame_flip[:, :, 1] = 255 - frame_flip[:, :, 1]
                    frame = frame - param.mean_value
                    frame_flip = frame_flip - param.mean_value
                    frame = np.concatenate((frame[:, :, :, np.newaxis], frame_flip[:, :, :, np.newaxis]), axis=3)
                else:
                    frame = frame - param.mean_value
                    frame = frame[:, :, :, np.newaxis]
                # python is column first
                frame = np.transpose(frame, (3, 2, 0, 1))
                # crop frames
                frame_data = crop(frame, param)

                if param.data_name not in data:
                    [l, c, h, w] = frame_data.shape
                    data[param.data_name] = np.zeros([l*fpv, c*length, h, w])
                data[param.data_name][x*l:(x+1)*l, i*c:(i+1)*c, :, :] = frame_data
    return data


def batch_forward(data, net, param):
    total_num = data[param.data_param[0].data_name].shape[0]
    batch_size = net.blobs[param.outputs[0]].num
    res = {}
    for output_name in param.outputs:
        out_name = output_name.encode('ascii', 'replace')  # savemat does not support unicode
        res[out_name] = np.zeros([total_num] + list(net.blobs[output_name].data.shape[1:]))
    for i in xrange(0, total_num, batch_size):
        idx = np.arange(i, min(i+batch_size,total_num))
        for input_name in (dp.data_name for dp in param.data_param):
            input_shape = [len(idx)] + list(data[input_name].shape[1:])
            net.blobs[input_name].reshape(*input_shape)
            net.blobs[input_name].data[...] = data[input_name][idx, :, :, :]
        out = net.forward()
        for output_name in out:
            res[output_name][idx, :] = out[output_name]
    return res


def classify(param, worker_id, nworkers, file_locs):
    # read test file
    fid = open(file_locs['split'], 'r')
    table = np.loadtxt(fid, delimiter=' ',
                       dtype={'names': ('files','nframes','cid'),
                              'formats': ('S100', 'i4', 'i4')})
    fid.close()
    # init network
    if param.env == two_stream.TestConfiguration.LOCAL:
        gpu_id = param.gpu_id[worker_id]
    else:
        gpu_id = 0

    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(file_locs['model'], file_locs['weights'], caffe.TEST)
    # run testing
    if not os.path.exists(file_locs['results']):
        os.makedirs(file_locs['results'])

    for i in xrange(len(table)):
        if i % nworkers == worker_id:
            fn = file_locs['results'] + '/' + table[i][0].split('/')[1] + '.mat'
            if not os.path.isfile(fn) or param.overwrite:
                # prepare data
                start = time.time()
                data = prepare_data(table[i][0], table[i][1], param.data_param)
                time_data = time.time() - start
                # batch forward
                res = batch_forward(data, net, param)
                time_all = time.time() - start
                print '%s: #%d %s...%.2f (%.2f) seconds' % \
                        (param.net_id, i, table[i][0].split('/')[1], time_all, time_data)
                scipy.io.savemat(fn, {'res':res})


def evaluate(file_locs):
    fid = open(file_locs['split'], 'r')
    table = np.loadtxt(fid, delimiter=' ',
                       dtype={'names': ('files', 'nframes', 'cid'),
                              'formats': ('S100', 'i4', 'i4')})
    fid.close()
    num_cls = len(np.unique(table['cid']))
    conf_mat = np.zeros((num_cls, num_cls))
    out = [[]]*len(table)
    for i in xrange(len(table)):
        fn = file_locs['results']+'/'+table['files'][i].split('/')[1]+'.mat'
        res = scipy.io.loadmat(fn)
        res = res['res']['prob'][0][0]
        out[i] = np.average(res,axis=0)
        conf_mat[table['cid'][i]][np.argmax(out[i])] += 1
    scipy.io.savemat(file_locs['results']+'.mat', {'out':out, 'conf_mat':conf_mat})

    overall = np.sum(np.diagonal(conf_mat))*1.0 / len(table)
    average = np.mean(np.divide(np.diagonal(conf_mat)*1.0, np.sum(conf_mat,axis=1)))
    print 'overall: %.3f, average: %.3f' % (overall, average)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Testing Action CNN')
    parser.add_argument('proto_txt', metavar='proto_txt', help='protobuf file for test configuration')
    parser.add_argument('phase', metavar='phase', help='phase of test (test/evaluate)',
                        choices=['test','evaluate'], default='test')
    parser.add_argument('--worker_id', dest='worker_id', help='id of the worker',
                        default=0, type=int)
    parser.add_argument('--nworkers', dest='nworkers', help='total number of workers',
                        default=1, type=int)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    param = two_stream.TestConfiguration()
    proto_txt = open(args.proto_txt, 'r')
    Merge(proto_txt.read(), param)
    proto_txt.close()

    suffix_method = []
    file_locs = {
        'cid' : ('%s/list/%s/classInd.txt' % (param.exper_name, param.net_id)).encode('ascii', 'replace'),
        'split' : ('%s/list/%s/test_UCF101_frames_split01.txt' % (param.exper_name, param.net_id)).encode('ascii', 'replace'),
        'model' : ('%s/config/%s/deploy.prototxt' % (param.exper_name, param.net_id)).encode('ascii', 'replace'),
        'weights' : ('%s/model/%s/%s.caffemodel' % (param.exper_name, param.net_id, param.model_name)).encode('ascii','replace')
    }
    for data_param in param.data_param:
        suffix = 's[' + data_param.sample_method
        if not data_param.sample_method == "all":
            suffix += '_fpv%d' % data_param.fpv
        suffix += ']'
        if not data_param.crop_method == "none":
            suffix += '_c[{0}_{1}x{2}]'.format(data_param.crop_method,
                                               data_param.crop_size[0],
                                               data_param.crop_size[1])
        suffix_method.append(suffix)

    result_dir = '%s/results/%s/%s' % (param.exper_name, param.net_id, '_'.join(suffix_method))
    if param.data_param[0].flip:
        result_dir += '_flip'
    file_locs['results'] = result_dir

    print 'class index file: ' + file_locs['cid']
    print 'test list file: ' + file_locs['split']
    print 'model definition file: ' + file_locs['model']
    print 'weights file: ' + file_locs['weights']
    print 'result dir: ' + file_locs['results']

    if args.phase == 'test':
        if param.env == two_stream.TestConfiguration.LOCAL:
            # if on local machine, use the provided gpu list
            nworkers = len(param.gpu_id)
        else:
            # on cluster, submit n jobs, each as one worker
            nworkers = args.nworkers
        classify(param, args.worker_id, nworkers, file_locs)
    else:
        evaluate(file_locs)

if __name__ == '__main__':
    sys.exit(main())
