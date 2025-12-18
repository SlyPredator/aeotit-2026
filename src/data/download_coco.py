from roboflow import Roboflow

rf = Roboflow(api_key="api/key/from/roboflow")
project = rf.workspace("image-ai-development").project("crack-segmentation-ryckv")
version = project.version(1)
dataset = version.download("coco")