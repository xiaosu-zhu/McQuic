添加 `NCCL_P2P_LEVEL=NVL` 环境变量提升显卡通信效率

quantizer 部分去除了 einsum，这会导致 grad stride 与 bucket view 不一致

由于level太多，rans coder不能用
