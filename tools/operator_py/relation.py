import mxnet as mx


def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
    # position_mat, [num_rois, nongt_dim, 4]
    feat_range = mx.sym.arange(0, feat_dim / 8)
    dim_mat = mx.sym.broadcast_power(lhs=mx.sym.full((1,), wave_length),
                                     rhs=(8. / feat_dim) * feat_range)
    dim_mat = mx.sym.Reshape(dim_mat, shape=(1, 1, 1, -1))
    position_mat = mx.sym.expand_dims(100.0 * position_mat, axis=3)
    div_mat = mx.sym.broadcast_div(lhs=position_mat, rhs=dim_mat)
    sin_mat = mx.sym.sin(data=div_mat)
    cos_mat = mx.sym.cos(data=div_mat)
    # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
    embedding = mx.sym.concat(sin_mat, cos_mat, dim=3)
    # embedding, [num_rois, nongt_dim, feat_dim]
    embedding = mx.sym.Reshape(embedding, shape=(0, 0, feat_dim))
    return embedding


def extract_position_matrix(bbox, nongt_dim):
    xmin, ymin, xmax, ymax = mx.sym.split(data=bbox,
                                          num_outputs=4, axis=1)
    # [num_fg_classes, num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [num_fg_classes, num_boxes, num_boxes]
    delta_x = mx.sym.broadcast_minus(lhs=center_x,
                                     rhs=mx.sym.transpose(center_x))
    delta_x = mx.sym.broadcast_div(delta_x, bbox_width)
    delta_x = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_x), 1e-3))
    delta_y = mx.sym.broadcast_minus(lhs=center_y,
                                     rhs=mx.sym.transpose(center_y))
    delta_y = mx.sym.broadcast_div(delta_y, bbox_height)
    delta_y = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_y), 1e-3))
    delta_width = mx.sym.broadcast_div(lhs=bbox_width,
                                       rhs=mx.sym.transpose(bbox_width))
    delta_width = mx.sym.log(delta_width)
    delta_height = mx.sym.broadcast_div(lhs=bbox_height,
                                        rhs=mx.sym.transpose(bbox_height))
    delta_height = mx.sym.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        sym = mx.sym.slice_axis(sym, axis=1, begin=0, end=nongt_dim)
        concat_list[idx] = mx.sym.expand_dims(sym, axis=2)
    position_matrix = mx.sym.concat(*concat_list, dim=2)
    return position_matrix



class RelationOperator(mx.operator.CustomOp):
    def __init__(self, nongt_dim):
        super(RelationOperator, self).__init__()
        self.nongt_dim = nongt_dim

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0]
        sliced_rois = mx.sym.slice_axis(rois, axis=1, begin=1, end=None)
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