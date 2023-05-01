from transformers import CLIPProcessor, CLIPModel
import torch

import numpy as np

class ClipClassifier:
    def __init__(self, device, labels=['hummingbird', 'not a bird']) -> None:
        model_id = "openai/clip-vit-large-patch14"
        print(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)

        # if you have CUDA set it to the active device like this
        self.device = device
        # move the model to the device
        self.model.to(self.device)

        self.labels= labels
        clip_labels = [f"a {label}" for label in self.labels]

        # create label tokens
        label_tokens = self.processor(
            text=clip_labels,
            padding=True,
            images=None,
            return_tensors='pt'
        ).to(self.device)

        # encode tokens to sentence embeddings
        label_emb = self.model.get_text_features(**label_tokens)
        # detach from pytorch gradient computation
        label_emb = label_emb.detach().cpu().numpy()
        # normalization
        self.label_emb = label_emb / np.linalg.norm(label_emb, axis=0)
    
    
    def img_preporcess_clip(self, image):
        return self.processor(
            text=None,
            images=image,
            return_tensors='pt'
            )['pixel_values'].to(self.device)
    
    
    def get_patches(self, img_tensor, patch_size=256):
        img_tensor = torch.squeeze(img_tensor, 0)
        patches = img_tensor.data.unfold(0,3,3)
        #horizontal
        patches = patches.unfold(1, patch_size, patch_size)
        # vertical
        patches = patches.unfold(2, patch_size, patch_size)
        return patches
    
    def get_scores(self, img_patches, prompt, window=6, stride=1, patch_size=256):
        
        scores = torch.zeros(img_patches.shape[1], img_patches.shape[2])
        runs = torch.ones(img_patches.shape[1], img_patches.shape[2])

        for Y in range(0, img_patches.shape[1]-window+1, stride):
            for X in range(0, img_patches.shape[2]-window+1, stride):
                big_patch = torch.zeros(patch_size*window, patch_size*window, 3)
                patch_batch = img_patches[0, Y:Y+window, X:X+window]
                for y in range(window):
                    for x in range(window):
                        big_patch[
                            y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size, :
                        ] = patch_batch[y, x].permute(1, 2, 0)
                # we preprocess the image and class label with the CLIP processor
                inputs = self.processor(
                    images=big_patch,  # big patch image sent to CLIP
                    return_tensors="pt",  # tell CLIP to return pytorch tensor
                    text=prompt,  # class label sent to CLIP
                    padding=True
                ).to(self.device) # move to device if possible

                # calculate and retrieve similarity score
                score = self.model(**inputs).logits_per_image.item()
                # sum up similarity scores from current and previous big patches
                # that were calculated for patches within the current window
                scores[Y:Y+window, X:X+window] += score
                # calculate the number of runs on each patch within the current window
                runs[Y:Y+window, X:X+window] += 1
        scores /= runs
        # clip the scores
        scores = np.clip(scores-scores.mean(), 0, np.inf)
        # normalize scores
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores
    
    def get_box(self, scores, patch_size, threshold=0.5):
        # scores higher than threshold are positive
        detection = scores > threshold
        y_min, y_max = (
            np.nonzero(detection)[:,0].min().item(),
            np.nonzero(detection)[:,0].max().item()+1
        )
        x_min, x_max = (
            np.nonzero(detection)[:,1].min().item(),
            np.nonzero(detection)[:,1].max().item()+1
        )
        y_min *= patch_size
        y_max *= patch_size
        x_min *= patch_size
        x_max *= patch_size
       
        return y_min,x_min, y_max, x_max

    def detect(self, img, patch_size=256, window=6, stride=1, threshold=0.7):
        # build image patches for detection
        img_patches = self.get_patches(img, patch_size)
        bboxes_scores = []
        # process image through object detection steps
        for idx, prompt in enumerate(self.labels):
            scores = self.get_scores(img_patches, prompt, window, stride, patch_size)
            x, y, width, height = self.get_box(scores, patch_size, threshold)
            bboxes_scores.append(([[x,y,width,height, 1,idx]]))
        return torch.from_numpy(np.array(bboxes_scores)).to(self.device)