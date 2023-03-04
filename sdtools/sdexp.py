import os
import pickle
import hashlib
from typing import Dict, List, Union, Optional
from diffusers import StableDiffusionPipeline
from PIL import Image
import clip
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import sdtools.finetune as finetune

def classhash(class_prompt:str):
    return hashlib.sha256(class_prompt.encode("utf-8")).hexdigest()

def run_exp(spec:Dict, verbose:bool=False):
    #example spec:
    # spec = {
    #     "exp":"A1",
    #     "entities":[
    #         {
    #             "finetune_path":"/home/vmisra/sd_exp/data/entitygirl",
    #             "class_prompt":"a cell phone photo of a girl",
    #             "finetune_prompt":"a cell phone photo of qaeks girl",
    #             "n_class_img":200
    #         }
    #     ],
    #     "lr":2e-6,
    #     "n_iters":[200, 400, 800, 1200, 1600, 2400],
    #     "dir_model":"/home/vmisra/sd_exp/data/modelexp/A1",
    #     "dir_parent_classimg":"/home/vmisra/sd_exp/data/classimg",
    #     "test_prompts":
    #         [
    #             "qaeks girl sits in a cornfield, smiling. Watercolor.",
    #             "qaeks girl licks an ice cream cone on a bench. Pencil drawing.",
    #             "qaeks girl hides in the corner, scared.",
    #             "qaeks girl lies in bed, dreaming about numbers. Digital art.",
    #             "qaeks girl sings into a microphone on stage, closeup."
    #         ]
    # }
    finetune.train_model(
        lst_dir_finetune_img = [entity['finetune_path'] for entity in spec['entities']],
        dir_model = spec['dir_model'],
        lst_prompt_finetune=[entity['finetune_prompt'] for entity in spec['entities']],
        lst_prompt_class=[entity['class_prompt'] for entity in spec['entities']],
        lst_n_classimg=[entity['n_class_img'] for entity in spec['entities']],
        lst_dir_class_img=[os.path.join(spec['dir_parent_classimg'],classhash(entity['class_prompt'])) for entity in spec['entities']],
        lr=spec['lr'],
        snapshot_steps=spec['n_iters'],
        verbose=verbose
    )

def load_exp(spec:Dict, iters:Optional[int]=None):
    if iters is None:
        iters:int = spec['n_iters'][-1]
    else:
        assert(iters in spec['n_iters'])
    
    dir_model = os.path.join(spec['dir_model'],f'model-iters-{iters}')
    assert(os.path.exists(dir_model))
    pipe = StableDiffusionPipeline.from_pretrained(dir_model, torch_dtype=torch.float16).to("cuda")

    return pipe

def sample_exp(spec:Dict, iters:Optional[int]=None, n_samples:int=10, lst_prompt = None):
    pipe = load_exp(spec, iters)
    lst_images=[]
    if lst_prompt is None:
        lst_prompt = spec['test_prompts']
    for prompt in lst_prompt:
        for idx in range(n_samples):
            lst_images.append(pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0])
    del pipe
    return lst_images
    

def eval_model(spec:Dict, iters:Optional[int]=None, n_examples:Optional[int]=10):
    pipe = load_exp(spec, iters)
    for prompt in spec['test_prompts']:
        print('-'*20)
        print(prompt)
        for idx in range(n_examples):
            print(idx)
            lst_image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images
            display(lst_image[0])
    del pipe

CLIP_BATCH_SIZE = 64
TEST_FRAC = 0.2
class CLFQuality():
    def __init__(self, lst_img:List[Image.Image], lst_instance_labels:List[int], lst_prompt_labels:List[int], path_cache:str, force_retrain:bool=False):
        self.clipmodel, self.clippreprocess = clip.load("ViT-B/32", device="cuda")
        if os.path.exists(path_cache) and not force_retrain:
            self.load(path_cache)
        else:
            self.fit(lst_img, lst_instance_labels, lst_prompt_labels)
            self.save(path_cache)
    
    def fit(self, lst_img:List[Image.Image], lst_instance_labels:List[int], lst_prompt_labels:List[int]):        
        #preprocess the images into a single tensor
        t_img = torch.cat([self.clippreprocess(img).unsqueeze(0) for img in lst_img], dim=0)

        #batch the images into CLIP_BATCH_SIZE, move them to the GPU, and get the image features out into numpy format. Wrap the whole thing in a torch.no_grad() context to save memory
        with torch.no_grad():
            lst_arr_img_feat = [self.clipmodel.encode_image(t_img[i*CLIP_BATCH_SIZE:(i+1)*CLIP_BATCH_SIZE].cuda()).cpu().numpy() for i in range(int(t_img.shape[0]/CLIP_BATCH_SIZE)+1)]
        #concatenate the image features into a single tensor
        arr_img_feat = np.concatenate(lst_arr_img_feat, axis=0)

        #split image features and all the labels into train and test sets
        n_test = int(TEST_FRAC*arr_img_feat.shape[0])
        arr_img_feat_train = arr_img_feat[:-n_test]
        arr_img_feat_test = arr_img_feat[-n_test:]
        lst_instance_labels_train = lst_instance_labels[:-n_test]
        lst_instance_labels_test = lst_instance_labels[-n_test:]
        lst_prompt_labels_train = lst_prompt_labels[:-n_test]
        lst_prompt_labels_test = lst_prompt_labels[-n_test:]

        #train a logistic regression model to predict the 0/1 instance labels and the 0/1 prompt labels from the image features
        clf_instance_test = LogisticRegression(random_state=0, max_iter=1000).fit(arr_img_feat_train, lst_instance_labels_train)
        clf_prompt_test = LogisticRegression(random_state=0, max_iter=1000).fit(arr_img_feat_train, lst_prompt_labels_train)

        #eval and print just for sanity checking.
        print(f"instance accuracy: {clf_instance_test.score(arr_img_feat_test, lst_instance_labels_test)}")
        print(f"prompt accuracy: {clf_prompt_test.score(arr_img_feat_test, lst_prompt_labels_test)}")

        #train the actual model for inferential use cases on the full labeled dataset.
        self.clf_instance = LogisticRegression(random_state=0).fit(arr_img_feat, lst_instance_labels)
        self.clf_prompt = LogisticRegression(random_state=0).fit(arr_img_feat, lst_prompt_labels)
    
    def load(self, path_cache:str):
        with open(path_cache,'rb') as f:
            self.clf_instance, self.clf_prompt = pickle.load(f)
    
    def save(self, path_cache:str):
        with open(path_cache,'wb') as f:
            pickle.dump((self.clf_instance, self.clf_prompt), f)
    
    def predict_proba(self, lst_img:List[Image.Image]):
        t_img = torch.cat([self.clippreprocess(img).unsqueeze(0) for img in lst_img], dim=0)
        with torch.no_grad():
            lst_arr_img_feat = [self.clipmodel.encode_image(t_img[i*CLIP_BATCH_SIZE:(i+1)*CLIP_BATCH_SIZE].cuda()).cpu().numpy() for i in range(int(t_img.shape[0]/CLIP_BATCH_SIZE)+1)]
        arr_img_feat = np.concatenate(lst_arr_img_feat, axis=0)
        return self.clf_instance.predict_proba(arr_img_feat), self.clf_prompt.predict_proba(arr_img_feat)