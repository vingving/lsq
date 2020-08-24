

# LSQ-Net

This is the implementation of Learned Step Size Quantization (LSQ) paper, ICLR2020
## Training

Use the following command to start training with the model is quantized into 4-bit weights and 4-bit activations

```bash
python main.py  --epochs 120 --lr 0.01  --bit 4  \
--dataset cifar10  --init_from checkpoint/cifar10_resnet_fp32.pth   --wd 1e-4
```
## Results

### CIFAR-10
[August 10] lr_scheduler in the epoch training-level, for the first convolutional layer, the weights are quantized only, leaving the input activations in FP32. Weights and activations in the last fully connected layer are both quantized.





![alt text][logo]

[logo]: http://i.imgur.com/eGmmGcb.png

### ImageNet

| Models | Networks   | Params      | Notes | Acc@1 |
|-----------------|------|-------------------|--------|------------|
| FP32      | Standard ResNet  | epochs=120,<br /> wd=1e-4,<br /> batch_size=512 |  Pytorch Model Zoo | 69.76 |
| W4A4      | Standard ResNet  | epochs=120,<br /> wd=1e-4,<br /> batch_size=512| The **weights** in the first layer is quantized to 4-bit, while **the incomming input data** is kept in FP32<br />  The *weights and activations* in the last layer is both quantized to 4-bit. <br/> Ref: *987b8d1bebf8cf1a2a7b00a702815f0a*  | 70.38 |
| W4A4      | Standard ResNet  | epochs=120,<br /> wd=1e-4,<br /> batch_size=512| The **weights** in the first layer is quantized to 8-bit, while **the incomming input data** is kept in FP32<br />  The *weights and activations* in the last layer is both quantized to 8-bit. |  Waiting [Uranas] |
| W4A4      | Standard ResNet  | epochs=120,<br /> wd=1e-4,<br /> batch_size=512| The **weights and activations** in the first layer  & last layer are both quantized to 8-bit. <br/> Ref: *4d28d6cb6bcfe23f499b484dcb9d48c6*  | 70.46  |
| W4A4      | Standard ResNet  | epochs=90,<br /> wd=1e-4,<br /> batch_size=512 |  The **weights and activations** in the first layer  & last layer are both quantized to 8-bit. <br/> Ref: *4b3354f59651539e1f238edfaa930e83* | 70.33 |

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)