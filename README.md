# equiEPNN

Based on source code for the paper "**[A Canonicalization Perspective on Invariant and Equivariant Learning](https://openreview.net/forum?id=jjcY92FX4R&noteId=jjcY92FX4R)**", NeurIPS 2024, [[SignNet repo](https://github.com/cptq/SignNet-BasisNet)] by Lim et al. in 2022, which in turn builds off of the setup in [[LSPE repo](https://github.com/vijaydwivedi75/gnn-lspe)] by Dwivedi et al. in 2021.

If you use our code, please cite

We want to run the following:

`
python main_OGBMOL_graph_classification.py  --gpu_id 0 --config 'configs/GatedGCN_MOLPCBA_OAP.json'
`

It automatically runs it on different seeds. We will get a mean with std.


Environment Setup:

mamba create -n canon-ogb-check python=3.8.10 -y
mamba activate canon-ogb-check 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -y
pip install numpy scipy networkx ogb packaging
pip install torchdata==0.6.1
pip install pydantic six
pip install ogb==1.3.6
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install tensorboardX matplotlib
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torch-geometric

