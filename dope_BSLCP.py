# Copyright 2020-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

import sys, os
import argparse
import os.path as osp
from PIL import Image
import cv2
import numpy as np
import pickle
from pathlib import Path

from collections import defaultdict

import torch
from torchvision.transforms import ToTensor

import gc

_thisdir = osp.realpath(osp.dirname(__file__))

from model import dope_resnet50, num_joints
import postprocess

import visu

def torch_to_list(torch_tensor):
    return torch_tensor.cpu().numpy().tolist()


class Dope():
    def __init__(self, video_path, modelname, postprocessing='ppi'):
        self.video_path = video_path
        self.modelname = modelname
        self.postprocessing = postprocessing
        self.load_model()


    def load_model(self):
        if self.postprocessing=='ppi':
            sys.path.append( _thisdir+'/lcrnet-v2-improved-ppi/')
            try:
                global LCRNet_PPI_improved
                from lcr_net_ppi_improved import LCRNet_PPI_improved
            except ModuleNotFoundError:
                raise Exception('To use the pose proposals integration (ppi) as self.postprocessing, please follow the readme instruction by cloning our modified version of LCRNet_v2.0 here. Alternatively, you can use --postprocess nms without any installation, with a slight decrease of performance.')

    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load model
        ckpt_fname = osp.join(_thisdir, 'models', self.modelname+'.pth.tgz')
        if not os.path.isfile(ckpt_fname):
            raise Exception('{:s} does not exist, please download the model first and place it in the models/ folder'.format(ckpt_fname))
        print('Loading model', self.modelname)
        self.ckpt = torch.load(ckpt_fname)
        #ckpt['half'] = False # uncomment this line in case your device cannot handle half computation
        self.ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000
        model = dope_resnet50(**self.ckpt['dope_kwargs'])
        if self.ckpt['half']: model = model.half()
        model = model.eval()
        model.load_state_dict(self.ckpt['state_dict'])
        self.model = model.to(self.device)


    def dope_videos(self):
        path_ending = os.path.join(*self.video_path.parts[7:])
        path_ending = Path(str(path_ending).replace('videos','Dope')).parent
        save_path = Path(f'/users/katrin/{path_ending}')
        save_path.mkdir(exist_ok=True, parents=True)

        path_ending2 = os.path.join(*self.video_path.parts[7:])
        path_ending2 = Path(str(path_ending2).replace('videos','DopeImages')).parent
        self.save_images_path = Path(f'/users/katrin/{path_ending2}/{Path(self.video_path).stem}')
        self.save_images_path.mkdir(exist_ok=True, parents=True) 
        
        # with open(self.info_file, 'rb') as f:
        #     info_data = pickle.load(f)

        # for ix, video in enumerate(info_data['videos']['name']):

        #     video_path = Path(self.video_folder) / Path(video)

        cap = cv2.VideoCapture(str(self.video_path))#'/users/katrin/coding/libs/segmentation/bsltrain/data/BSLCP/videos/Conversation/Belfast/1+2/BF1c.mov'))
        frame_nr = 0
        result = defaultdict(lambda: {})

        list_cuda = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections, results = self.dope_single_image(frame, frame_nr)
                frame_nr += 1
                result[frame_nr]['det'] = detections
                result[frame_nr]['res'] = results

        cap.release()
        with open(f'{save_path}/{Path(self.video_path).stem}_Dope.pkl', "wb") as f:
            print(f"Saving to {save_path}/{Path(self.video_path).stem}_Dope.pkl")
            pickle.dump(result, f)


    

    def dope_single_image(self, image, frame_nr=0):
            
        # load the image
        # print('Loading image', imagename)
        # image = Image.open(imagename)
        imlist = [ToTensor()(image).to(self.device)]
        if self.ckpt['half']: imlist = [im.half() for im in imlist]
        resolution = imlist[0].size()[-2:]
        
        # forward pass of the dope network
        #print('Running DOPE')
        with torch.no_grad():
            results = self.model(imlist, None)[0]

        
        # postprocess results (pose proposals integration, wrists/head assignment)
        #print('Postprocessing')
        assert self.postprocessing in ['nms','ppi']
        parts = ['body','hand','face']
        if self.postprocessing=='ppi':
            res = {k: v.float().data.cpu().numpy() for k,v in results.items()}
            detections = {}
            for part in parts:
                detections[part] = LCRNet_PPI_improved(res[part+'_scores'], res['boxes'], res[part+'_pose2d'], res[part+'_pose3d'], resolution, **self.ckpt[part+'_ppi_kwargs'])
        else: # nms
            detections = {}
            for part in parts:
                dets, indices, bestcls = postprocess.DOPE_NMS(results[part+'_scores'], results['boxes'], results[part+'_pose2d'], results[part+'_pose3d'], min_score=0.3)
                dets = {k: v.float().data.cpu().numpy() for k,v in dets.items()}
                detections[part] = [{'score': dets['score'][i], 'pose2d': dets['pose2d'][i,...], 'pose3d': dets['pose3d'][i,...]} for i in range(dets['score'].size)]
                if part=='hand':
                    for i in range(len(detections[part])): 
                        detections[part][i]['hand_isright'] = bestcls<self.ckpt['hand_ppi_kwargs']['K']
        # assignment of hands and head to body
        detections = postprocess.assign_hands_and_head_to_body(detections)
        
        # display results
        if frame_nr%200 == 0:
            print('Displaying')
            det_poses2d = {part: np.stack([d['pose2d'] for d in part_detections], axis=0) if len(part_detections)>0 else np.empty( (0,num_joints[part],2), dtype=np.float32) for part, part_detections in detections.items()}
            scores = {part: [d['score'] for d in part_detections] for part,part_detections in detections.items()}
            imout = visu.visualize_bodyhandface2d(np.asarray(image)[:,:,::-1],
                                                det_poses2d,
                                                dict_scores=scores,
                                                )
            outfile = f'{self.save_images_path}/{frame_nr}.jpg'
            cv2.imwrite(outfile, imout)
            print(outfile)

        results_cpu = {}
        for x,y in results.items():
            results_cpu[x] = torch_to_list(y)

        return detections, results_cpu
        




if __name__=="__main__":

    parser = argparse.ArgumentParser(description='running DOPE on an image: python dope.py --model <modelname>')
    parser.add_argument('--model', default='DOPErealtime_v1_0_0', type=str, help='name of the model to use (eg DOPE_v1_0_0)')
    parser.add_argument('--postprocess', default='ppi', choices=['ppi','nms'], help='postprocessing method')
    parser.add_argument('--video_path', default='/users/katrin/coding/libs/segmentation/bsltrain/data/BSLCP/videos/Narrative/Cardiff/21+22/CF22n.mov')#required=True)


    # parser.add_argument('--info_file', default='/users/katrin/data/BSLCP/info/BSLCP_consecutive/info_cut_hist_10_clean_Katrin_merge_resize.pkl')
    # parser.add_argument('--video_folder', default='/users/katrin/data/BSLCP/videos-resized-25fps-256x256-BSLCP_consecutive_glosses_cut_hist_10_clean_Katrin_merge_resize')


    args = parser.parse_args()

    dope = Dope(Path(args.video_path), args.model, postprocessing=args.postprocess)
    dope.dope_videos()

    #print('Loading image', imagename)
    # image = Image.open('/users/katrin/coding/libs/dope/test_DOPErealtime_v1_0_0.jpg') 
    # dope.dope_single_image(image)


    # dope = Dope(args.info_file, args.video_folder, args.model, postprocessing=args.postprocess)
    # dope.dope_videos()
