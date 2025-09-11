from segment_anything import sam_model_registry, SamPredictor
from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoImageProcessor, AutoModel
from utils.dreamsim.dreamsim import dreamsim
# from archieve import SAM_ROOT, OWL_ROOT, CACHE_ROOT

SAM_ROOT = '/data/model/segment-anything/sam_vit_h_4b8939.pth'
OWL_ROOT = "google/owlv2-base-patch16-ensemble"
CACHE_ROOT = '/data/sangwon/multi_subject/utils/cache'

def load_sam(model_type = "vit_h", device='cuda:0'):
    sam = sam_model_registry[model_type](checkpoint=SAM_ROOT)
    return SamPredictor(sam.to(device))

def load_owl(device='cuda:0'):
    processor = Owlv2Processor.from_pretrained(OWL_ROOT)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_ROOT)
    return processor, model.to(device)

def load_dinov2(device='cuda:0'):
    preprocessor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    return preprocessor, model.to(device)

def load_dino(device='cuda:0'):
    preprocessor = AutoImageProcessor.from_pretrained('facebook/dino-vits16')
    model = AutoModel.from_pretrained('facebook/dino-vits16')
    return preprocessor, model.to(device)

def load_dreamsim(pretrained=True, cache_dir=CACHE_ROOT):
    return dreamsim(pretrained=pretrained, cache_dir=cache_dir)