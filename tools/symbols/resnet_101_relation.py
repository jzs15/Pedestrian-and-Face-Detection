import cPickle
import mxnet as mx
import numpy as np
from resnet_101 import ResNet101
from utils.symbol import Symbol
from operator_py.pyramid_proposal import *
from operator_py.proposal_target import *
from operator_py.fpn_roi_pooling import *
from operator_py.box_annotator_ohem import *
from operator_py.nms_multi_target import *
from operator_py.learn_nms import *


class ResNet101Relation(ResNet101):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.shared_param_list = ['rpn_conv', 'rpn_cls_score', 'rpn_bbox_pred']
        self.shared_param_dict = {}
        for name in self.shared_param_list:
            self.shared_param_dict[name + '_weight'] = mx.sym.Variable(name + '_weight')
            self.shared_param_dict[name + '_bias'] = mx.sym.Variable(name + '_bias')

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        res2, res3, res4, res5 = self.get_resnet_backbone(data)
        fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.get_fpn_feature(res2, res3, res4, res5)

        rpn_cls_score_p2, rpn_prob_p2, rpn_bbox_loss_p2, rpn_bbox_pred_p2 = self.get_rpn_subnet(fpn_p2,
                                                                                                cfg.network.NUM_ANCHORS,
                                                                                                'p2')
        rpn_cls_score_p3, rpn_prob_p3, rpn_bbox_loss_p3, rpn_bbox_pred_p3 = self.get_rpn_subnet(fpn_p3,
                                                                                                cfg.network.NUM_ANCHORS,
                                                                                                'p3')
        rpn_cls_score_p4, rpn_prob_p4, rpn_bbox_loss_p4, rpn_bbox_pred_p4 = self.get_rpn_subnet(fpn_p4,
                                                                                                cfg.network.NUM_ANCHORS,
                                                                                                'p4')
        rpn_cls_score_p5, rpn_prob_p5, rpn_bbox_loss_p5, rpn_bbox_pred_p5 = self.get_rpn_subnet(fpn_p5,
                                                                                                cfg.network.NUM_ANCHORS,
                                                                                                'p5')
        rpn_cls_score_p6, rpn_prob_p6, rpn_bbox_loss_p6, rpn_bbox_pred_p6 = self.get_rpn_subnet(fpn_p6,
                                                                                                cfg.network.NUM_ANCHORS,
                                                                                                'p6')

        rpn_cls_prob_dict = {
            'rpn_cls_prob_stride64': rpn_prob_p6,
            'rpn_cls_prob_stride32': rpn_prob_p5,
            'rpn_cls_prob_stride16': rpn_prob_p4,
            'rpn_cls_prob_stride8': rpn_prob_p3,
            'rpn_cls_prob_stride4': rpn_prob_p2,
        }
        rpn_bbox_pred_dict = {
            'rpn_bbox_pred_stride64': rpn_bbox_pred_p6,
            'rpn_bbox_pred_stride32': rpn_bbox_pred_p5,
            'rpn_bbox_pred_stride16': rpn_bbox_pred_p4,
            'rpn_bbox_pred_stride8': rpn_bbox_pred_p3,
            'rpn_bbox_pred_stride4': rpn_bbox_pred_p2,
        }
        arg_dict = dict(rpn_cls_prob_dict.items() + rpn_bbox_pred_dict.items())

        if is_train:
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            gt_boxes = mx.sym.Variable(name="gt_boxes")

            rpn_cls_score = mx.sym.Concat(rpn_cls_score_p2, rpn_cls_score_p3, rpn_cls_score_p4, rpn_cls_score_p5,
                                          rpn_cls_score_p6, dim=2)
            rpn_bbox_loss = mx.sym.Concat(rpn_bbox_loss_p2, rpn_bbox_loss_p3, rpn_bbox_loss_p4, rpn_bbox_loss_p5,
                                          rpn_bbox_loss_p6, dim=2)
            # RPN classification loss
            rpn_cls_output = mx.sym.SoftmaxOutput(data=rpn_cls_score, label=rpn_label, multi_output=True,
                                                  normalization='valid',
                                                  use_ignore=True, ignore_label=-1, name='rpn_cls_prob')
            # bounding box regression
            rpn_bbox_loss = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_l1', scalar=3.0,
                                                               data=(rpn_bbox_loss - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss,
                                            grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            aux_dict = {
                'op_type': 'pyramid_proposal', 'name': 'rois',
                'im_info': im_info, 'feat_stride': tuple(cfg.network.RPN_FEAT_STRIDE),
                'scales': tuple(cfg.network.ANCHOR_SCALES), 'ratios': tuple(cfg.network.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n': cfg.TRAIN.RPN_PRE_NMS_TOP_N, 'rpn_post_nms_top_n': cfg.TRAIN.RPN_POST_NMS_TOP_N,
                'threshold': cfg.TRAIN.RPN_NMS_THRESH, 'rpn_min_size': cfg.TRAIN.RPN_MIN_SIZE
            }

            # ROI proposal
            rois = mx.sym.Custom(**dict(arg_dict.items() + aux_dict.items()))
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight \
                = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                                num_classes=num_reg_classes, batch_images=cfg.TRAIN.BATCH_IMAGES,
                                batch_rois=cfg.TRAIN.BATCH_ROIS, cfg=cPickle.dumps(cfg),
                                fg_fraction=cfg.TRAIN.FG_FRACTION)
        else:
            aux_dict = {
                'op_type': 'pyramid_proposal', 'name': 'rois',
                'im_info': im_info, 'feat_stride': tuple(cfg.network.RPN_FEAT_STRIDE),
                'scales': tuple(cfg.network.ANCHOR_SCALES), 'ratios': tuple(cfg.network.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n': cfg.TEST.RPN_PRE_NMS_TOP_N, 'rpn_post_nms_top_n': cfg.TEST.RPN_POST_NMS_TOP_N,
                'threshold': cfg.TEST.RPN_NMS_THRESH, 'rpn_min_size': cfg.TEST.RPN_MIN_SIZE
            }
            # ROI proposal
            rois = mx.sym.Custom(**dict(arg_dict.items() + aux_dict.items()))
        nongt_dim = cfg.TRAIN.RPN_POST_NMS_TOP_N if is_train else cfg.TEST.RPN_POST_NMS_TOP_N
        roi_pool = mx.symbol.Custom(data_p2=fpn_p2, data_p3=fpn_p3, data_p4=fpn_p4, data_p5=fpn_p5,
                                    rois=rois, op_type='fpn_roi_pooling', name='fpn_roi_pooling')

        sliced_rois = mx.sym.slice_axis(rois, axis=1, begin=1, end=None)
        # [num_rois, nongt_dim, 4]
        position_matrix = self.extract_position_matrix(sliced_rois, nongt_dim=nongt_dim)
        # [num_rois, nongt_dim, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)

        # 2 fc
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=1024)
        attention_1 = self.attention_module_multi_head(fc_new_1, position_embedding,
                                                       nongt_dim=nongt_dim, fc_dim=16, feat_dim=1024,
                                                       index=1, group=16,
                                                       dim=(1024, 1024, 1024))
        fc_new_1 = fc_new_1 + attention_1
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

        fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
        attention_2 = self.attention_module_multi_head(fc_new_2, position_embedding,
                                                       nongt_dim=nongt_dim, fc_dim=16, feat_dim=1024,
                                                       index=2, group=16,
                                                       dim=(1024, 1024, 1024))
        fc_new_2 = fc_new_2 + attention_2
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')
        # cls_score/bbox_pred
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

        if is_train:
            labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                           num_reg_classes=num_reg_classes,
                                                           roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                           cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                           bbox_targets=bbox_target, bbox_weights=bbox_weight)
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                            normalization='valid', use_ignore=True, ignore_label=-1)
            bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                              data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                        grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
            rcnn_label = labels_ohem

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            # group = mx.sym.Group([rpn_cls_output, rpn_bbox_loss, mx.sym.BlockGrad(cls_prob), mx.sym.BlockGrad(bbox_loss), mx.sym.BlockGrad(rcnn_label)])
            group = [rpn_cls_output, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)]
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred_reshape = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = [rois, cls_prob, bbox_pred_reshape]

        ######################### learn nms #########################
        # notice that all implementation of python ops try to leave batch idx support for multi-batch
        # thus, rois are [batch_ind, x_min, y_min, x_max, y_max]
        nms_target_thresh = np.fromstring(cfg.network.NMS_TARGET_THRESH, dtype=float, sep=',')
        num_thresh = len(nms_target_thresh)
        nms_eps = 1e-8
        first_n = cfg.TRAIN.FIRST_N if is_train else cfg.TEST.FIRST_N
        num_fg_classes = num_classes - 1
        bbox_means = cfg.TRAIN.BBOX_MEANS if is_train else None
        bbox_stds = cfg.TRAIN.BBOX_STDS if is_train else None
        nongt_dim = cfg.TRAIN.RPN_POST_NMS_TOP_N if is_train else cfg.TEST.RPN_POST_NMS_TOP_N

        if is_train:
            # remove gt here
            cls_score_nongt = mx.sym.slice_axis(data=cls_score, axis=0, begin=0, end=nongt_dim)
            bbox_pred_nongt = mx.sym.slice_axis(data=bbox_pred, axis=0, begin=0, end=nongt_dim)
            bbox_pred_nongt = mx.sym.BlockGrad(bbox_pred_nongt)

            # refine bbox
            # remove batch idx and gt roi
            sliced_rois = mx.sym.slice(data=rois, begin=(0, 1), end=(nongt_dim, None))
            # bbox_pred_nobg, [num_rois, 4*(num_reg_classes-1)]
            bbox_pred_nobg = mx.sym.slice_axis(data=bbox_pred_nongt, axis=1, begin=4, end=None)
            # [num_boxes, 4, num_reg_classes-1]
            refined_bbox = self.refine_bbox(sliced_rois, bbox_pred_nobg, im_info,
                                            means=bbox_means, stds=bbox_stds
                                            )
            # softmax cls_score to cls_prob, [num_rois, num_classes]
            cls_prob = mx.sym.softmax(data=cls_score_nongt, axis=-1)
            cls_prob_nobg = mx.sym.slice_axis(cls_prob, axis=1, begin=1, end=None)
            sorted_cls_prob_nobg = mx.sym.sort(data=cls_prob_nobg, axis=0, is_ascend=False)
            # sorted_score, [first_n, num_fg_classes]
            sorted_score = mx.sym.slice_axis(sorted_cls_prob_nobg, axis=0,
                                             begin=0, end=first_n, name='sorted_score')
            # sort by score
            rank_indices = mx.sym.argsort(data=cls_prob_nobg, axis=0, is_ascend=False)
            # first_rank_indices, [first_n, num_fg_classes]
            first_rank_indices = mx.sym.slice_axis(rank_indices, axis=0, begin=0, end=first_n)
            # sorted_bbox, [first_n, num_fg_classes, 4, num_reg_classes-1]
            sorted_bbox = mx.sym.take(a=refined_bbox, indices=first_rank_indices)
            if cfg.CLASS_AGNOSTIC:
                # sorted_bbox, [first_n, num_fg_classes, 4]
                sorted_bbox = mx.sym.Reshape(sorted_bbox, shape=(0, 0, 0), name='sorted_bbox')
            else:
                cls_mask = mx.sym.arange(0, num_fg_classes)
                cls_mask = mx.sym.Reshape(cls_mask, shape=(1, -1, 1))
                cls_mask = mx.sym.broadcast_to(cls_mask, shape=(first_n, 0, 4))
                # sorted_bbox, [first_n, num_fg_classes, 4]
                sorted_bbox = mx.sym.pick(data=sorted_bbox, name='sorted_bbox',
                                          index=cls_mask, axis=3)
            # nms_rank_embedding, [first_n, 1024]
            nms_rank_embedding = self.extract_rank_embedding(first_n, 1024)
            # nms_rank_feat, [first_n, 1024]
            nms_rank_feat = mx.sym.FullyConnected(name='nms_rank', data=nms_rank_embedding, num_hidden=128)
            # nms_position_matrix, [num_fg_classes, first_n, first_n, 4]
            nms_position_matrix = self.extract_multi_position_matrix(sorted_bbox)
            # roi_feature_embedding, [num_rois, 1024]
            roi_feat_embedding = mx.sym.FullyConnected(
                name='roi_feat_embedding',
                data=fc_new_2_relu,
                num_hidden=128)
            # sorted_roi_feat, [first_n, num_fg_classes, 128]
            sorted_roi_feat = mx.sym.take(a=roi_feat_embedding, indices=first_rank_indices)

            # vectorized nms
            # nms_embedding_feat, [first_n, num_fg_classes, 128]
            nms_embedding_feat = mx.sym.broadcast_add(
                lhs=sorted_roi_feat,
                rhs=mx.sym.expand_dims(nms_rank_feat, axis=1))
            # nms_attention_1, [first_n, num_fg_classes, 1024]
            nms_attention_1, nms_softmax_1 = self.attention_module_nms_multi_head(
                nms_embedding_feat, nms_position_matrix,
                num_rois=first_n, index=1, group=16,
                dim=(1024, 1024, 128), fc_dim=(64, 16), feat_dim=128)
            nms_all_feat_1 = nms_embedding_feat + nms_attention_1
            nms_all_feat_1_relu = mx.sym.Activation(data=nms_all_feat_1, act_type='relu', name='nms_all_feat_1_relu')
            # [first_n * num_fg_classes, 1024]
            nms_all_feat_1_relu_reshape = mx.sym.Reshape(nms_all_feat_1_relu, shape=(-3, -2))
            # logit, [first_n * num_fg_classes, num_thresh]
            nms_conditional_logit = mx.sym.FullyConnected(name='nms_logit',
                                                          data=nms_all_feat_1_relu_reshape,
                                                          num_hidden=num_thresh)
            # logit_reshape, [first_n, num_fg_classes, num_thresh]
            nms_conditional_logit_reshape = mx.sym.Reshape(nms_conditional_logit,
                                                           shape=(first_n, num_fg_classes, num_thresh))
            nms_conditional_score = mx.sym.Activation(data=nms_conditional_logit_reshape,
                                                      act_type='sigmoid', name='nms_conditional_score')
            sorted_score_reshape = mx.sym.expand_dims(sorted_score, axis=2)
            # sorted_score_reshape = mx.sym.BlockGrad(sorted_score_reshape)
            nms_multi_score = mx.sym.broadcast_mul(lhs=sorted_score_reshape, rhs=nms_conditional_score)
        else:
            nms_rank_weight = mx.sym.var('nms_rank_weight', shape=(128, 1024), dtype=np.float32)
            nms_rank_bias = mx.sym.var('nms_rank_bias', shape=(128,), dtype=np.float32)
            roi_feat_embedding_weight = mx.sym.var('roi_feat_embedding_weight', shape=(128, 1024), dtype=np.float32)
            roi_feat_embedding_bias = mx.sym.var('roi_feat_embedding_bias', shape=(128,), dtype=np.float32)
            nms_pair_pos_fc1_1_weight = mx.sym.var('nms_pair_pos_fc1_1_weight', shape=(16, 64), dtype=np.float32)
            nms_pair_pos_fc1_1_bias = mx.sym.var('nms_pair_pos_fc1_1_bias', shape=(16,), dtype=np.float32)
            nms_query_1_weight = mx.sym.var('nms_query_1_weight', shape=(1024, 128), dtype=np.float32)
            nms_query_1_bias = mx.sym.var('nms_query_1_bias', shape=(1024,), dtype=np.float32)
            nms_key_1_weight = mx.sym.var('nms_key_1_weight', shape=(1024, 128), dtype=np.float32)
            nms_key_1_bias = mx.sym.var('nms_key_1_bias', shape=(1024,), dtype=np.float32)
            nms_linear_out_1_weight = mx.sym.var('nms_linear_out_1_weight', shape=(128, 128, 1, 1), dtype=np.float32)
            nms_linear_out_1_bias = mx.sym.var('nms_linear_out_1_bias', shape=(128,), dtype=np.float32)
            nms_logit_weight = mx.sym.var('nms_logit_weight', shape=(5, 128), dtype=np.float32)
            nms_logit_bias = mx.sym.var('nms_logit_bias', shape=(5,), dtype=np.float32)

            nms_multi_score, sorted_bbox, sorted_score = mx.sym.Custom(cls_score=cls_score, bbox_pred=bbox_pred,
                                                                       rois=rois, im_info=im_info,
                                                                       nms_rank_weight=nms_rank_weight,
                                                                       fc_new_2_relu=fc_new_2_relu,
                                                                       nms_rank_bias=nms_rank_bias,
                                                                       roi_feat_embedding_weight=roi_feat_embedding_weight,
                                                                       roi_feat_embedding_bias=roi_feat_embedding_bias,
                                                                       nms_pair_pos_fc1_1_weight=nms_pair_pos_fc1_1_weight,
                                                                       nms_pair_pos_fc1_1_bias=nms_pair_pos_fc1_1_bias,
                                                                       nms_query_1_weight=nms_query_1_weight,
                                                                       nms_query_1_bias=nms_query_1_bias,
                                                                       nms_key_1_weight=nms_key_1_weight,
                                                                       nms_key_1_bias=nms_key_1_bias,
                                                                       nms_linear_out_1_weight=nms_linear_out_1_weight,
                                                                       nms_linear_out_1_bias=nms_linear_out_1_bias,
                                                                       nms_logit_weight=nms_logit_weight,
                                                                       nms_logit_bias=nms_logit_bias,
                                                                       op_type='learn_nms', name='learn_nms',
                                                                       num_fg_classes=num_fg_classes,
                                                                       bbox_means=bbox_means, bbox_stds=bbox_stds,
                                                                       first_n=first_n,
                                                                       class_agnostic=cfg.CLASS_AGNOSTIC,
                                                                       num_thresh=num_thresh,
                                                                       class_thresh=cfg.TEST.LEARN_NMS_CLASS_SCORE_TH,
                                                                       nongt_dim=nongt_dim, has_non_gt_index=False)

        if is_train:
            nms_multi_target = mx.sym.Custom(bbox=sorted_bbox, gt_bbox=gt_boxes, score=sorted_score,
                                             op_type='nms_multi_target', target_thresh=nms_target_thresh)
            nms_pos_loss = - mx.sym.broadcast_mul(lhs=nms_multi_target,
                                                  rhs=mx.sym.log(data=(nms_multi_score + nms_eps)))
            nms_neg_loss = - mx.sym.broadcast_mul(lhs=(1.0 - nms_multi_target),
                                                  rhs=mx.sym.log(data=(1.0 - nms_multi_score + nms_eps)))
            normalizer = first_n * num_thresh
            nms_pos_loss = cfg.TRAIN.nms_loss_scale * nms_pos_loss / normalizer
            nms_neg_loss = cfg.TRAIN.nms_loss_scale * nms_neg_loss / normalizer
            ##########################  additional output!  ##########################
            group.append(mx.sym.BlockGrad(nms_multi_target, name='nms_multi_target_block'))
            group.append(mx.sym.BlockGrad(nms_conditional_score, name='nms_conditional_score_block'))
            group.append(mx.sym.MakeLoss(name='nms_pos_loss', data=nms_pos_loss,
                                                   grad_scale=cfg.TRAIN.nms_pos_scale))
            group.append(mx.sym.MakeLoss(name='nms_neg_loss', data=nms_neg_loss))
        else:
            if cfg.TEST.MERGE_METHOD == -1:
                nms_final_score = mx.sym.mean(data=nms_multi_score, axis=2, name='nms_final_score')
            elif cfg.TEST.MERGE_METHOD == -2:
                nms_final_score = mx.sym.max(data=nms_multi_score, axis=2, name='nms_final_score')
            elif 0 <= cfg.TEST.MERGE_METHOD < num_thresh:
                idx = cfg.TEST.MERGE_METHOD
                nms_final_score = mx.sym.slice_axis(data=nms_multi_score, axis=2, begin=idx, end=idx + 1)
                nms_final_score = mx.sym.Reshape(nms_final_score, shape=(0, 0), name='nms_final_score')
            else:
                raise NotImplementedError('Unknown merge method %s.' % cfg.TEST.MERGE_METHOD)
            group.append(sorted_bbox)
            group.append(sorted_score)
            group.append(nms_final_score)

        self.sym = mx.sym.Group(group)
        # mx.viz.plot_network(self.sym, title='graph', save_format='svg', hide_weights=True).view()
        return self.sym

    def init_weight_attention_nms_multi_head(self, cfg, arg_params, aux_params, index=1):
        arg_params['nms_pair_pos_fc1_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_pair_pos_fc1_' + str(index) + '_weight'])
        arg_params['nms_pair_pos_fc1_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_pair_pos_fc1_' + str(index) + '_bias'])
        arg_params['nms_query_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_query_' + str(index) + '_weight'])
        arg_params['nms_query_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_query_' + str(index) + '_bias'])
        arg_params['nms_key_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_key_' + str(index) + '_weight'])
        arg_params['nms_key_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_key_' + str(index) + '_bias'])
        arg_params['nms_linear_out_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_linear_out_' + str(index) + '_weight'])
        arg_params['nms_linear_out_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_linear_out_' + str(index) + '_bias'])

    def init_weight_nms(self, cfg, arg_params,aux_params):
        arg_params['nms_rank_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_rank_weight'])
        arg_params['nms_rank_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['nms_rank_bias'])
        arg_params['roi_feat_embedding_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['roi_feat_embedding_weight'])
        arg_params['roi_feat_embedding_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['roi_feat_embedding_bias'])
        self.init_weight_attention_nms_multi_head(cfg, arg_params, aux_params, index=1)
        arg_params['nms_logit_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_logit_weight'])
        arg_params['nms_logit_bias'] = mx.nd.full(shape=self.arg_shape_dict['nms_logit_bias'], val=-3.0)

    def init_weight(self, cfg, arg_params, aux_params):
        for name in self.shared_param_list:
            arg_params[name + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[name + '_weight'])
            arg_params[name + '_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[name + '_bias'])
        self.init_weight_rcnn(cfg, arg_params, aux_params)
        self.init_weight_fpn(cfg, arg_params, aux_params)
        self.init_weight_nms(cfg, arg_params, aux_params)

