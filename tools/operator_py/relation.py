import mxnet as mx
import numpy as np

def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
    pos_shape = position_mat.shape
    feat_range = np.arange(0, feat_dim / 8)
    dim_mat = np.power(np.full((1,), wave_length), (8. / feat_dim) * feat_range)
    dim_mat = np.reshape(dim_mat, (1, 1, 1, -1))
    position_mat = np.expand_dims(100.0 * position_mat, axis=3)
    div_mat = position_mat / dim_mat
    sin_mat = np.sin(div_mat)
    cos_mat = np.cos(div_mat)
    embedding = np.concatenate([sin_mat, cos_mat], axis=3)
    embedding = np.reshape(embedding, (pos_shape[0], pos_shape[1], feat_dim))
    return embedding


def extract_position_matrix(bbox, nongt_dim):
    xmin, ymin, xmax, ymax = np.split(bbox, 4, 1)
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [num_fg_classes, num_boxes, num_boxes]
    delta_x = center_x - center_x.T
    delta_x = delta_x / bbox_width
    delta_x = np.log(np.maximum(np.abs(delta_x), 1e-3))
    delta_y = center_y - center_y.T
    delta_y = delta_y / bbox_height
    delta_y = np.log(np.maximum(np.abs(delta_y), 1e-3))
    delta_width = bbox_width / bbox_width.T
    delta_width = np.log(delta_width)
    delta_height = bbox_height / bbox_height.T
    delta_height = np.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        sym = sym[:, :nongt_dim]
        concat_list[idx] = np.expand_dims(sym, axis=2)
    position_matrix = np.concatenate(concat_list, axis=2)
    return position_matrix



class RelationOperator(mx.operator.CustomOp):
    def __init__(self, nongt_dim):
        super(RelationOperator, self).__init__()
        self.nongt_dim = nongt_dim

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0].asnumpy()
        sliced_rois = rois[:, 1:]
        position_matrix = extract_position_matrix(sliced_rois, nongt_dim=self.nongt_dim)
        position_embedding = extract_position_embedding(position_matrix, feat_dim=64)
        self.assign(out_data[0], req[0], position_embedding)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)



@mx.operator.register('relation')
class RelationProp(mx.operator.CustomOpProp):
    def __init__(self, nongt_dim):
        super(RelationProp, self).__init__(need_top_grad=False)
        self.nongt_dim = int(nongt_dim)

    def list_arguments(self):
        return ['rois']

    def list_outputs(self):
        return ['position_matrix']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        position_matrix_shape = (rois_shape[0], self.nongt_dim, 64)

        return in_shape, [position_matrix_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return RelationOperator(self.nongt_dim)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []