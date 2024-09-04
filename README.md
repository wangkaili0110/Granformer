# Granformer
This project is the PyTorch implementation of "Granformer:a granular transformer net with linear complexity".[[paper]](https://doi.org/10.1016/j.neucom.2024.128380)

## Introduction
Granformer is a granular transformer framework with linear complexity by using granular attention and linearization of matrix factorization. 
It makes the feature representation performance of Transformer more accurate and efficient.

![Granformer](./data/illustration.png)  

### Dependencies
Ubuntu == 20.04  
GPU == NVIDIA A100  
GPU Driver == 535.104.05  
CUDA == 12.2  
Python == 3.8  
Pytorch == 2.2.0  
Torchvision == 0.17.0

```shell script
pip install pointnet2_ops_lib/.
pip install -r requirements.txt
```
The code is from https://github.com/wangkaili0110/Granformer

### Accuracy
ModelNet40:
<table>
  <thead>
    <tr style="text-align: center;">
      <th>model</th>
      <th>input</th>
      <th>input size</th>
      <th>OA(%)</th>
      <th>mA(%)</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>Granformer-Gaussian</td>
      <td>P</td>
      <td>1024*3</td>
      <td>93.3</td>
      <td>90.6</td>
    </tr>
    <tr>
      <td>Granformer-Laplacian</td>
      <td>P</td>
      <td>1024*3</td>
      <td>93.6</td>
      <td>90.1</td>
    </tr>
    <tr>
      <td>Granformer-Multivariate quadratic</td>
      <td>P</td>
      <td>1024*3</td>
      <td>93.4</td>
      <td>90.6</td>
    </tr>
    <tr>
      <td>Granformer-Neighborhood</td>
      <td>P</td>
      <td>1024*3</td>
      <td>93.3</td>
      <td>90.3</td>
    </tr>
    <tr>
      <td>Granformer-Partial</td>
      <td>P</td>
      <td>1024*3</td>
      <td>93.6</td>
      <td>90.9</td>
    </tr>
  </tbody>
</table>

CMU-MOSEI:
<table>
  <thead>
    <tr style="text-align: center;">
      <th>method</th>
      <th>Sentiment-2(Acc%)</th>
      <th>Sentiment-7(Acc%)</th>
      <th>Emotion-6(Acc%)</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>Granformer-Gaussian</td>
      <td>82.47</td>
      <td>45.37</td>
      <td>81.57</td>
    </tr>
    <tr>
      <td>Granformer-Laplacian</td>
      <td>82.45</td>
      <td>45.38</td>
      <td>81.52</td>
    </tr>
    <tr>
      <td>Granformer-Multivariate quadratic</td>
      <td>82.41</td>
      <td>45.36</td>
      <td>81.54</td>
    </tr>
    <tr>
      <td>Granformer-Neighborhood</td>
      <td>82.43</td>
      <td>45.37</td>
      <td>81.57</td>
    </tr>
    <tr>
      <td>Granformer-Partial</td>
      <td>82.46</td>
      <td>45.39</td>
      <td>81.57</td>
    </tr>
  </tbody>
</table>

COCO:
<table>
  <thead>
    <tr style="text-align: center;">
      <th>model</th>
      <th>Ap</th>
      <th>Ap50</th>
      <th>Ap75</th>
      <th>Aps</th>
      <th>Apm</th>
      <th>Apl</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>Granformer-Laplacian</td>
      <td>38.3</td>
      <td>57.9</td>
      <td>40.8</td>
      <td>21.1</td>
      <td>42.2</td>
      <td>52.7</td>
    </tr>
  </tbody>
</table>

### Citation
If you find Granformer useful in your research, please consider citing:
```bibtex
@article{wang2024granformer,
      title={Granformer:a granular transformer net with linear complexity}, 
      author={Kaili Wang and Xinwei Sun and Tao Shen},
      journal={Neurocomputing},
      url={https://doi.org/10.1016/j.neucom.2024.128380},
      year={2024}
}
```
