'''
    nuscenes-specified occam
    
'''
import torch
import argparse
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from occam_utils.occam import OccAM
import numpy as np

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_pp_multihead.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default="/home/yzy/occam/pp_multihead_nds5823_updated.pth", help='specify the pretrained model')
    parser.add_argument('--occam_cfg_file', type=str,
                        default='cfgs/occam_configs/kitti_pointpillar.yaml',
                        help='specify the OccAM config')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for OccAM creation')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for dataloader')
    parser.add_argument('--nr_it', type=int, default=6000,
                        help='number of sub-sampling iterations N')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_from_yaml_file(args.occam_cfg_file, cfg)
    return args, cfg


def get_base_prediction_nus(data_dict):
    '''
    Get prediction results of nuscenes data
    '''    
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('---------Demo of occam_nus-------------')
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=None, workers=4,
        logger=logger,
        training=False,
        merge_all_iters_to_one_epoch=False,
        total_epochs=None
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(train_set):
            data_dict = train_set.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            base_det_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            base_det_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            base_det_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pcl = data_dict['points']
    base_det_boxes = base_det_boxes[:, :7]
    
    return base_det_boxes, base_det_labels, base_det_scores

def main():
    args, config = parse_config()
    logger = common_utils.create_logger()
    logger.info('------------------------ OccAM Demo -------------------------')

    occam = OccAM(data_config=config.DATA_CONFIG, model_config=config.MODEL,
                  occam_config=config.OCCAM, class_names=config.CLASS_NAMES,
                  model_ckpt_path=args.ckpt, nr_it=args.nr_it, logger=logger)

    '''
    data_dict()?
        为每个点云数据构建一次data_dict，pcl根据data_dict进行提取
    '''    
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=None, workers=4,
        logger=logger,
        training=False,
        merge_all_iters_to_one_epoch=False,
        total_epochs=None
    )
    idx = 0
    base_det = get_base_prediction_nus(train_set[idx])
    base_det_boxes, base_det_labels, base_det_scores = base_det
    print('base_det_boxes = ', base_det_boxes.shape)
    print('base_det_labels = ', base_det_labels.shape[0])
    print('base_det_scores = ', base_det_scores.shape[0])
    print(base_det_labels)
    logger.info('Number of detected objects to analyze: '
                + str(base_det_labels.shape[0]))
    base_det_boxes = base_det_boxes[:, :7]
    print(base_det_boxes.shape)
    logger.info('Start attribution map computation:')
    pcl = train_set[idx]['points'][:, 1:]
    attr_maps = occam.compute_attribution_maps(
        pcl=pcl, base_det_boxes=base_det_boxes,
        base_det_labels=base_det_labels, batch_size=args.batch_size,
        num_workers=args.workers)

    logger.info('DONE')

    save_path = '/home/yzy/occam/for_vis/pointpillars_multihead/'
    np.save(save_path + 'attr_maps', attr_maps)
    np.save(save_path + 'box', base_det_boxes)
    np.save(save_path + 'labels', base_det_labels)
    np.save(save_path + 'pcl', pcl)
    # logger.info('Visualize attribution map of first object')
    #occam.visualize_attr_map(pcl, base_det_boxes[0, :], attr_maps[0, :])
    #取第一个检测，和他对应的bounding_box


if __name__ == '__main__':
    main()
