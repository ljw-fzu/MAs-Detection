model = dict(
    type='FasterSSPNet', # model类型
    pretrained='torchvision://resnet50', # 预训练模型：imagenet-resnet50
    backbone=dict(
        type='ResNet',# backbone类型
        depth=50,    # 网络层数
        num_stages=4,   # resnet的stage数量
        out_indices=(0, 1, 2, 3),  # 输出的stage的序号
        frozen_stages=1,  # 冻结的stage数量，即该stage不更新参数，-1表示所有的stage都更新参数
        norm_cfg=dict(type='BN', requires_grad=True), # 网络风格：如果设置pytorch，则stride为2的层是conv3x3的卷积层；如果设置caffe，则stride为2的层是第一个conv1x1的卷积层
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='SSFPN',    # neck类型
        in_channels=[256, 512, 1024, 2048], # 输入的各个stage的通道数
        out_channels=256,  # 输出的特征层的通道数
        num_outs=5), # 输出的特征层的数量
    rpn_head=dict(
        type='RPNHead',  # RPN网络类型
        in_channels=256, # RPN网络的输入通道数
        feat_channels=256,    # 特征层的通道数
        anchor_generator=dict(
            type='AnchorGenerator',
            #ma anchor########################、
            # #e_o
            # scales=[3, 5, 7, 10],   # 生成的anchor的baselen（面积的开方），baselen = sqrt(w*h)，w和h为anchor的宽和高
            # ratios=[0.4, 1, 1.5], # anchor的宽高比  h / w  原

            #idrid
            scales=[1.5, 2.5, 3.7, 8],   # 生成的anchor的baselen（面积的开方），baselen = sqrt(w*h)，w和h为anchor的宽和高
            ratios=[0.4, 0.8, 1.4], # anchor的宽高比  h / w  原

            # #ROC
            # scales=[10, 15, 17, 20],  # 生成的anchor的baselen（面积的开方），baselen = sqrt(w*h)，w和h为anchor的宽和高
            # ratios=[0.5, 1.0, 1.5],  # anchor的宽高比  h / w
            ########################################################
            # scales=[1.5, 2.5, 3.7, 8],   # 生成的anchor的baselen（面积的开方），baselen = sqrt(w*h)，w和h为anchor的宽和高
            # # ratios=[0.4, 1.0, 1.4], # anchor的宽高比  h / w
            # ratios=[0.4, 0.8, 1.4],  # anchor的宽高比  h / w

            strides=[4, 8, 16, 32, 64]),  # 在每个特征层上的anchor的步长（对应于原图）如果不对应将导致图片部分区域没有anchor或anchor的设置超过图片边缘,strides严格等于下采样倍数即可
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],    # 均值
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # 方差
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            #WNS策略
            # type='ICNegSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_threshold=0.3, min_score=0.05),
        max_per_img=500))
