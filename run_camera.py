import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import time
from torchvision.transforms import Compose
from pythonosc import udp_client


from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

if __name__ == '__main__':
    
  
    encoder = 'vits' 
    video_path = 1   

    # OSC setting
    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 6448)

    # use different calculate method
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(" Running on: CUDA (GTX 950M)")
    else:
        DEVICE = 'cpu'
        print(" Running on: CPU")
    
    
    print("Loading Model...")
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    

    target_size = 196 
    
    transform = Compose([
        Resize(
            width=448,
            height=294,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    cap = cv2.VideoCapture(video_path)
    

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 448)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 294)
    
   
    font = cv2.FONT_HERSHEY_SIMPLEX

    last_mean_depth = 0
    
    frame_count = 0
    SKIP_FRAMES = 2  
    

    cached_features = [0, 0, 0, 0, 0.5, 0.5] 
    cached_depth_vis = None


    while cap.isOpened():
        ret, raw_image = cap.read()
        if not ret: break

        frame_count += 1
        
     
        if frame_count % (SKIP_FRAMES + 1) == 0:
           
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            h, w = image.shape[:2]

            image_tensor = transform({'image': image})['image']
            image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                depth = depth_anything(image_tensor)
            
         
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            
         
            raw_depth = depth.cpu().numpy()
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(raw_depth)
            norm_depth = (raw_depth - min_val) / (max_val - min_val + 1e-6)

            mean_depth = float(np.mean(raw_depth))
            occupancy_rate = float(np.sum(norm_depth > 0.7) / (h * w))
            variance = float(np.var(raw_depth))
            
            delta_depth = float(abs(mean_depth - last_mean_depth))
            last_mean_depth = mean_depth 

            focus_x = float(max_loc[0] / w)
            focus_y = float(max_loc[1] / h)

            cached_features = [mean_depth, occupancy_rate, variance, delta_depth, focus_x, focus_y]
            
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            cached_depth_vis = depth_vis.cpu().numpy().astype(np.uint8)

      
        osc_client.send_message("/depth/features", cached_features)
        
       
        if cached_depth_vis is not None:
            depth_color = cv2.applyColorMap(cached_depth_vis, cv2.COLORMAP_INFERNO)
         
            fx = int(cached_features[4] * raw_image.shape[1])
            fy = int(cached_features[5] * raw_image.shape[0])
            cv2.circle(depth_color, (fx, fy), 10, (0, 255, 0), 2)
            
          
            combined = cv2.hconcat([raw_image, depth_color])
            cv2.imshow('Fast Depth', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()