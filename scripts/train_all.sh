RAQN-Replication/
├── env/                         # 网络环境（Mininet + Ryu）
│   ├── topology.py              # GEANT拓扑生成脚本
│   ├── controller.py            # Ryu控制器逻辑（带RAQN接口）
├── rl/
│   ├── model.py                 # RAQN神经网络结构
│   ├── train.py                 # 训练主流程（含引导样本机制）
│   ├── aco_lb_guidance.py       # 虚拟环境中ACO和LB算法样本生成
│   ├── replay_buffer.py         # Replay Buffer（含指导样本池）
│   ├── config.py                # 所有超参数统一配置
├── data/
│   ├── traffic_gen.py           # 流量生成脚本（iperf + 高斯分布）
│   ├── packet_analysis.py       # tshark分析脚本提取吞吐、延迟、丢包
├── visualize/
│   ├── plot_metrics.py          #
├── main.py                      # 训练主入口
├── requirements.txt             # 所需Python依赖
└── README.md

