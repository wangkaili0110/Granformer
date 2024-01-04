## PCT: Point Cloud Transformer
This is a Pytorch implementation of Granformer: Point Cloud Granformer.

Paper link: https://XXXX.pdf

### Requirements
python >= 3.7

pytorch >= 1.6

h5py

scikit-learn

and

```shell script
pip install pointnet2_ops_lib/.
```
The code is from https://github.com/wangkaili0110/Granformer

### Models
We get an accuracy of 93.6% on the ModelNet40(http://modelnet.cs.princeton.edu/) validation dataset

The path of the model is in ./checkpoints/best/models/model.t7

### Example training and testing
```shell script
# train
python main.py --exp_name=train --num_points=1024 --use_sgd=True --batch_size 32 --epochs 250 --lr 0.0001

# test
python main.py --exp_name=test --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/best/models/model.t7 --test_batch_size 8

```

### Citation
If it is helpful for your work, please cite this paper:
```latex
@misc{guo2020pct,
      title={Granformer: Point Cloud Granformer}, 
      author={Kai-Li Wang and Xin-Wei Sun and Tao Shen},
      year={2023},
      eprint={xxxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
