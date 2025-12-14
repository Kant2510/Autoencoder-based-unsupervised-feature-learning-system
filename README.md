```
nvcc -std=c++17 test.cu models/layers/Conv2D.cu models/layers/MaxPool2D.cu models/tensor.cu models/layers/Relu.cu models/layers/Upsample2D.cu models/autoencoder.cu optimizer/loss.cu utils/logs.cu dataset/dataset.cu kernels/optimizer/loss.cu kernels/optimizer/update_weight.cu kernels/conv2d.cu kernels/maxpool2d.cu kernels/relu.cu kernels/upsample2d.cu -o test.exe
test.exe
```
