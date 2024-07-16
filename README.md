# nvidia-gpu-arch
![image](https://github.com/ziwon/nvidia-gpu-arch/assets/152046/cd190350-7b60-41d6-8730-d41a68651aac)

## Overview
### GH200 
- Grace Hopper Superchip, Coherent NVLink
- Combines NVIDIA Grace CPU with Hopper GPU.
- Coherent NVLink, offering unified memory architecture and exceptional bandwidth.
- Up to 384GB HBM3e with 16 TB/s bandwidth.
- Enhanced for large-scale AI model training and inference, with significant energy efficiency and speed improvements.
- Decompression engine, multimedia decoders, and ARM Neoverse V2 cores for optimized data handling and retrieval​

### B200
- Blackwell Architecture, NVLink 5, 2024
- Higher performance than B100 with up to 9 PFLOPS for dense FP4 tensor operations and 18 PFLOPS for sparse FP4.
- Supports 192GB of HBM3e memory with 8 TB/s bandwidth.
- Enhanced NVLink and PCIe Gen6 for improved data transfer, and a TDP of 1000W, making it suitable for the most demanding AI tasks.
- Power: 1000W TDP, designed for high-end performance and demanding AI tasks​ 

### B100 
- Blackwell Architecture, NVLink 5, 2024
- Balanced computational efficiency with up to 7 PFLOPS for dense FP4 tensor operations and 14 PFLOPS for sparse FP4.
- Supports 192GB of HBM3e memory with 8 TB/s bandwidth.
- Utilizes NVLink for 1.8 TB/s bandwidth, suitable for high-performance AI and HPC workloads.
- Power: 700W TDP, suitable for energy-efficient setups​

### H200
- Hopper Architecture, NVLink and NVSwitch, 2024
- 141GB HBM3e memory with 4.8 TB/s bandwidth
- Enhanced for large language models (LLMs) and HPC.
- 1.6x faster inference performance for GPT-3, 1.9x faster for Llama2 70B.
- 50% lower energy requirements compared to H100.

### H100 
- Hopper Architecture, NVLink-C2C, 2022
- NVLink-C2C for direct GPU communication without CPU involvement.
- Advanced AI capabilities, especially for large-scale models.
- Enhanced Tensor Cores and Transformer Engine for efficiency.
- [H100 White Paper](https://jingchaozhang.github.io/H100-white-paper/)
![image](https://github.com/ziwon/nvidia-gpu-arch/assets/152046/a7a709a3-3cab-4f98-bc42-5bcd4359a96e)

### A100
- Ampere Architecture, NVLink 3, Transformer Engine, 2020
- Transformer Engine optimized for transformer networks.
- Multi-instance GPU (MIG) for better utilization.
- Improved Tensor Cores for enhanced AI model training and inference.
- [A100 White Paper](https://jingchaozhang.github.io/A100-white-paper/)
![image](https://github.com/ziwon/nvidia-gpu-arch/assets/152046/2d36f60e-42d6-404b-ba74-22c2f2b2eaea)

### V100 
- Volta Architecture, NVLink2, Tensor Core, 2017
- Introduction of Tensor Cores for accelerated deep learning tasks.
- Improved NVLink for faster GPU-to-GPU communication.
- Enhanced performance for AI and HPC workloads.

### P100 
- Pascal Architecture, NVLink, 2016
- NVLink for high-speed interconnect between GPUs and CPUs.
- Enhanced floating-point performance (FP64 and FP32).
- Focus on scientific computing, simulations, and deep learning.

## Docs
- [NVIDIA Blackwell Architecture](https://resources.nvidia.com/en-us-blackwell-architecture)
- [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core)
- [NVIDIA A100 Tensor Core GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA Turing GPU Architecture](https://www.nvidia.com/en-us/design-visualization/technologies/turing-architecture/)
- [NVIDIA TESLA V100](http://www.nvidia.com/object/volta-architecture-whitepaper.html)
- [NVIDIA TESLA P100](http://www.nvidia.com/object/gpu-architecture.html)
- [AMD CDNA2](https://www.amd.com/system/files/documents/amd-cdna2-white-paper.pdf)
- [AMD CDNA](https://www.amd.com/system/files/documents/amd-cdna-whitepaper.pdf)

- [B200 Datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)
- [H100 Datasheet](https://resources.nvidia.com/en-us-dgx-systems/ai-enterprise-dgx)
  
- [NVIDIA DGX SuperPOD: Next Generation Scalable Infrastructure for AI Leadership](https://docs.nvidia.com/https:/docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)
- [NVIDIA DGX BasePOD: The Infrastructure Foundation for Enterprise AI](https://resources.nvidia.com/en-us-dgx-systems/nvidia-dgx-basepod)

- [ConnectX-7 400G Adapters](https://nvdam.widen.net/s/srdqzxgdr5/connectx-7-datasheet)
- [ConnectX-6 200G Adapters](https://nvdam.widen.net/s/qpszhmhpzt/networking-overal-dpu-datasheet-connectx-6-dx-smartnic-1991450)
- [ConnectX-5 100G Adapters](https://network.nvidia.com/files/doc-2020/pb-connectx-5-en-card.pdf)

## User Guide
- [H100](https://docs.nvidia.com/dgx/dgxh100-user-guide/dgxh100-user-guide.pdf)
- [A100](/https://docs.nvidia.com/dgx/pdf/dgxa100-user-guide.pdf)

## Papers
- [GPU Domain Specialization via Composable On-Package Architecture](https://arxiv.org/abs/2104.02188)

## Articles
- [NVIDIA’s Blackwell Architecture: Breaking Down The B100, B200, and GB200](https://www.linkedin.com/pulse/nvidias-blackwell-architecture-breaking-down-b100-b200-gb200-wlp0c)
- [NVIDIA Blackwell Architecture and B200/B100 Accelerators Announced: Going Bigger With Smaller Data](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data)
- [Nvidia Blackwell Perf TCO Analysis - B100 vs B200 vs GB200NVL72](https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis)
- [Trends in Deep Learning Hardware: Bill Dally (NVIDIA)](https://www.youtube.com/watch?v=kLiwvnr4L80)
- [Hpc network technologies](https://jingchaozhang.github.io/HPC-network-technologies/)

## Certifications
- [(NCA-AIIO) AI Infrastructure and Operations](https://www.nvidia.com/en-us/learn/certification/ai-infrastructure-operations-associate/)
- [(NCA-GENL) Generative AI LLMs](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/)
- [(NCP-IB) InfiniBand](https://www.nvidia.com/en-us/learn/certification/infiniband-professional/)

## Networking
- [Evolution of Data Center Networking Designs and Systems for AI Infrastructure – Part 1](https://www.linkedin.com/pulse/evolution-data-center-networking-designs-systems-ai-part-sujal-das-obyec/)
- [Evolution of Data Center Networking Designs and Systems for AI Infrastructure – Part 2](https://www.linkedin.com/pulse/evolution-data-center-networking-designs-systems-ai-part-sujal-das-hshtc/)
- [Evolution of Data Center Networking Designs and Systems for AI Infrastructure – Part 3](https://www.linkedin.com/pulse/evolution-data-center-networking-designs-systems-ai-part-sujal-das-lne4c/)
- [Evolution of Data Center Networking Designs and Systems for AI Infrastructure – Part 4 (Final)](https://www.linkedin.com/pulse/evolution-data-center-networking-designs-systems-ai-part-sujal-das-hiauc)
- [GPU Fabrics for GenAI Workloads](https://www.linkedin.com/pulse/gpu-fabrics-genai-workloads-sharada-yeluri-j8ghc)
  - [Youtube](https://www.youtube.com/watch?v=lTrHzqZ8Imo) 

## Course
- [Deep Learning Fundamentals](https://lightning.ai/courses/deep-learning-fundamentals/)
- [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering)
- [Google - Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

## Ops
- [LLM Inference - HW/SW Optimizations](https://www.linkedin.com/pulse/llm-inference-hwsw-optimizations-sharada-yeluri-wfdyc)
- [Accelerating Self-Attentions for LLM Serving with FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)
- [A complete guide to AI accelerators for deep learning inference — GPUs, AWS Inferentia and Amazon Elastic Inference](https://towardsdatascience.com/a-complete-guide-to-ai-accelerators-for-deep-learning-inference-gpus-aws-inferentia-and-amazon-7a5d6804ef1c)
- [How Docker Runs Machine Learning on NVIDIA GPUs, AWS Inferentia, and Other Hardware AI Accelerators](https://towardsdatascience.com/how-docker-runs-machine-learning-on-nvidia-gpus-aws-inferentia-and-other-hardware-ai-accelerators-e076c6eb7802)
- [Understanding GPU Memory 1: Visualizing All Allocations over Time](https://pytorch.org/blog/understanding-gpu-memory-1/)
- [Understanding GPU Memory 2: Finding and Removing Reference Cycles](https://pytorch.org/blog/understanding-gpu-memory-2/)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
  - [Universal Checkpointing: Efficient and Flexible Checkpointing for Large Scale Distributed Training](https://arxiv.org/abs/2406.18820)
- [+Kyle’s excellent write-ups on testing distributed systems](https://jepsen.io/analyses)

## Tools
- [nsight compute](https://developer.nvidia.com/nsight-compute)
- [nvtop](https://github.com/Syllo/nvtop)
- [gtop](https://github.com/aksakalli/gtop)
- [gmond](https://github.com/ganglia/gmond_python_modules/tree/master/gpu/nvidia)
- [gpustat](https://github.com/wookayin/gpustat)
- [gpu-utils](https://github.com/Ricks-Lab/gpu-utils)
- [Statistics on GPUs](https://github.com/owensgroup/gpustats)

## Links
- [GPU Guide](https://github.com/mikeroyal/GPU-Guide)
- [NVIDIA Linux Open GPU Kernel Module Source](https://github.com/NVIDIA/open-gpu-kernel-modules)
