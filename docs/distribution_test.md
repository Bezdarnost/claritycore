# Training stategies - comparison

The tests was done during the training of [HAT](https://arxiv.org/abs/2309.05239) for x4 super resolution task with input size 3x64x64.

BasicSR uses the upstream implementation “as provided by the authors.”
Our ClarityCore runs enable several training optimizations (mixed BF16, `torch.compile` and other) while keeping architecture and accuracy identical.

For 8xRTX Pro 6000 Blackwell(96GB each):

| Framework   | Strategy       | Time for 100 iters (s) | Throughput (batches/s) | VRAM / GPU (GB) | GB / batch |
| ----------- | -------------- | ---------------------: | ---------------------: | --------------: | ---------: |
| BasicSR     | DDP            |                        |                        |                 |            |
| ClarityCore | DDP            |                        |                        |                 |            |
| ClarityCore | FSDP1          |                        |                        |                 |            |
| ClarityCore | FSDP2          |                        |                        |                 |            |
| ClarityCore | SimpleFSDP     |                        |                        |                 |            |
| ClarityCore | ZeRO-2         |                        |                        |                 |            |
| ClarityCore | ZeRO-3         |                        |                        |                 |            |

For 4xRTX Pro 6000 Blackwell(96GB each):

| Framework   | Strategy       | Time for 100 iters (s) | Throughput (batches/s) | VRAM / GPU (GB) | GB / batch |
| ----------- | -------------- | ---------------------: | ---------------------: | --------------: | ---------: |
| BasicSR     | DDP            |                        |                        |                 |            |
| ClarityCore | DDP            |                        |                        |                 |            |
| ClarityCore | FSDP1          |                        |                        |                 |            |
| ClarityCore | FSDP2          |                        |                        |                 |            |
| ClarityCore | SimpleFSDP     |                        |                        |                 |            |
| ClarityCore | ZeRO-2         |                        |                        |                 |            |
| ClarityCore | ZeRO-3         |                        |                        |                 |            |

For 2xRTX Pro 6000 Blackwell(96GB each):

| Framework   | Strategy       | Time for 100 iters (s) | Throughput (batches/s) | VRAM / GPU (GB) | GB / batch |
| ----------- | -------------- | ---------------------: | ---------------------: | --------------: | ---------: |
| BasicSR     | DDP            |                        |                        |                 |            |
| ClarityCore | DDP            |                        |                        |                 |            |
| ClarityCore | FSDP1          |                        |                        |                 |            |
| ClarityCore | FSDP2          |                        |                        |                 |            |
| ClarityCore | SimpleFSDP     |                        |                        |                 |            |
| ClarityCore | ZeRO-2         |                        |                        |                 |            |
| ClarityCore | ZeRO-3         |                        |                        |                 |            |

For 1xRTX Pro 6000 Blackwell(96GB):

| Framework   | Strategy       | Time for 100 iters (s) | Throughput (batches/s) | VRAM / GPU (GB) | GB / batch |
| ----------- | -------------- | ---------------------: | ---------------------: | --------------: | ---------: |
| BasicSR     | DDP            |                        |                        |                 |            |
| ClarityCore | DDP            |                        |                        |                 |            |
| ClarityCore | FSDP1          |                        |                        |                 |            |
| ClarityCore | FSDP2          |                        |                        |                 |            |
| ClarityCore | SimpleFSDP     |                        |                        |                 |            |
| ClarityCore | ZeRO-2         |                        |                        |                 |            |
| ClarityCore | ZeRO-3         |                        |                        |                 |            |
