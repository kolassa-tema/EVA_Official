pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -v -e preprocess/hamer/third-party/ViTPose
pip install third_party/diff-gaussian-rasterization
pip install third_party/simple-knn
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install third_party/torch-mesh-isect
pip install third_party/neural_renderer
pip install --no-deps git+https://github.com/nghorbani/human_body_prior.git@cvpr19
pip install git+https://github.com/facebookresearch/detectron2.git
pip install smplx onnxruntime plyfile lpips==0.1.4 configargparse
pip install pytorch-lightning==1.9.0 pyrender timm einops webdataset
pip install -r requirements.txt
pip install ninja


