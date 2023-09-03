# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2
import torch
from PIL import Image

def extract_image_patch(image, bbox, model_name='yolov8', patch_shape=None):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox.cpu())
    if model_name=='yolov8':
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None

    sx, sy, ex, ey = bbox
    
    image = image[sy:ey, sx:ex]

    #image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, model, transform, device):

        
        self.model = model
        self.transform = transform
        self.device = device
        self.imagenet_classes = []
        

    def __call__(self, data_x, batch_size=32):
        out = []
        #data_x = [i for i in data_x if i is not None]

        #print("[ZSOT ImageEncoder] num_none: {}".format(len(num_none)))
        for patch in range(len(data_x)):
            if self.device == "cpu":
                img = self.transform(Image.fromarray(data_x[patch]))
            else:
                img = self.transform(Image.fromarray(data_x[patch])).cuda()
            out.append(img)

        features = self.model.encode_image(torch.stack(out))
       
        features = features.cpu().numpy()
        for idx, i in enumerate(features):
            if np.isnan(i[0]):
                print("nan values")
                # features[idx] = np.zeros(512)
                # cv2.imshow("image", data_x[idx])
                cv2.waitKey(0)

        return features
    @staticmethod
    def argmax(iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]


def create_box_encoder(model, model_name = 'yolov8'):
    
    def encoder(image, boxes):
        clf_dict = []
        img_features = []
        for box in boxes:
            #print("extracting box {} from image {}".format(box, image.shape))
            patch = extract_image_patch(image, box, model_name=model_name)
            
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image.shape).astype(np.uint8)
                
            clip_patch = model.img_preporcess_clip(patch)

            img_emb = model.model.get_image_features(clip_patch).detach().cpu().numpy()
            clas_scores = np.dot(img_emb, model.label_emb.T)/(np.linalg.norm(img_emb)*np.linalg.norm(model.label_emb.T))
            
            clas_pred = np.argmax(clas_scores)
           
            img_emb = img_emb.squeeze()
            if clas_pred ==0 and clas_scores[0][clas_pred] >= 0.1:
                clf_dict.append({'label': model.labels[clas_pred],'score':clas_scores[0][clas_pred], 'img_features': img_emb })
            elif clas_pred ==0 and clas_scores[0][clas_pred] < 0.1:
                clf_dict.append({'label': model.labels[-1],'score':clas_scores[0][clas_pred], 'img_features': img_emb })
            else:
                clf_dict.append({'label': model.labels[clas_pred],'score':clas_scores[0][clas_pred], 'img_features': img_emb })

        img_features = np.array([x['img_features'] for x in clf_dict])
           
    
        return img_features, clf_dict

    return encoder

