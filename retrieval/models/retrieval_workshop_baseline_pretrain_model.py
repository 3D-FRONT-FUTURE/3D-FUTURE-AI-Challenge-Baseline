import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pdb
from .networks import Normalize
from .pytorch_metric_learning.pytorch_metric_learning import losses
import torch.nn as nn
import torch.nn.functional as F

class RetrievalWorkshopBaselinePretrainModel(BaseModel):
    """
    Pretrain   
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """
        model init
        """
        BaseModel.__init__(self, opt)
        self.pose_corr = 0.0
        self.center_corr = 0.0
        self.cate_corr = 0.0
        self.pose_center_corr = 0.0

        self.loss_names = ['query_center', 'query_cate', 'metric'] #, 'mask', 'G', 'D']
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['_backbone', '_further_conv', '_cate_estimator', '_center_estimator'] #, '_mask', '_discriminator']
        else:  # during test time, only load Gs
            self.model_names = ['_backbone', '_further_conv', '_cate_estimator', '_center_estimator'] #, '_mask']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.net_backbone = networks.define_retrieval_nets(opt, net_option='resnet34_pytorch', gpu_ids=self.gpu_ids)
        self.net_further_conv = networks.define_retrieval_nets(opt, net_option='further_conv', gpu_ids=self.gpu_ids)
        opt.input_dim = 256 # feature dims
        opt.cate_num = 5202 # instance classification, 5202 models, 5202 labels
        self.net_center_estimator = networks.define_retrieval_nets(opt, net_option='cate_estimator', gpu_ids=self.gpu_ids)
        opt.cate_num = 7    # leve2 cate classification
        self.net_cate_estimator = networks.define_retrieval_nets(opt, net_option='cate_estimator', gpu_ids=self.gpu_ids)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterionBce = torch.nn.BCEWithLogitsLoss()

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # define loss functions
            self.criterionSoftmax  = torch.nn.CrossEntropyLoss()
            self.criterionMetric = losses.TripletMarginLoss(triplets_per_anchor="all")
            self.criterionTriplet = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)
            self.optimizer = torch.optim.SGD(itertools.chain(self.net_backbone.parameters(), self.net_further_conv.parameters(), self.net_center_estimator.parameters(), self.net_cate_estimator.parameters()), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.input_query = input['query_img'].to(self.device)
        self.input_positive = input['positive_img'].to(self.device)
        self.input_negative = input['negative_img'].to(self.device)
        self.label_cate = input['cate_label'].type(torch.LongTensor).to(self.device)
        self.label_center = input['center_label'].type(torch.LongTensor).to(self.device)
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        query_backbone_feat = self.net_further_conv(self.net_backbone(self.input_query))
        # supervised classification
        self.cate_feat, self.query_cate_score = self.net_cate_estimator(query_backbone_feat, return_feat=True)
        self.query_center_score = self.net_center_estimator(query_backbone_feat)

    def backward(self):
        """Calculate the loss"""
        self.loss_query_cate = self.criterionSoftmax(self.query_cate_score, self.label_cate)*0.1
        self.loss_query_center = self.criterionSoftmax(self.query_center_score, self.label_center)*1.0
        self.loss_metric = self.criterionMetric(self.cate_feat, self.label_cate)*0.1

        self.loss = self.loss_query_cate + self.loss_query_center + self.loss_metric # + self.loss_mask + self.loss_G 

        _, max_cate = torch.max(self.query_cate_score.data, 1)
        cate_corr = torch.sum(max_cate == self.label_cate.data)
        self.cate_corr += cate_corr.item()
        
        _, max_center = torch.max(self.query_center_score.data, 1)
        center_corr = torch.sum(max_center == self.label_center.data)
        self.center_corr += center_corr.item()

        self.loss.backward()

   
    def set_input_eval(self, input):
        self.test_source = int(input['test_source'][0].numpy())
        if self.test_source == 0:
            self.pred_3d_data = input['query_img'].to(self.device)
        else:
            self.gt_3d_data = torch.squeeze(input['positive_img']).to(self.device)
    
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()              # compute fake images and reconstruction images.
        
        self.optimizer.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward()             # calculate gradients for G_A and G_B
        self.optimizer.step()       # update G_A and G_B's weights
        
