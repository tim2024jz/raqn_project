# RAQN Reproducibility Project

This project contains a full reproducible implementation of the RAQN algorithm as described in the paper.
It includes:
- GEANT topology using Mininet
- Ryu controller with RAQN logic
- RAQN & baseline algorithms
- Training & evaluation scripts
- Metrics collection and visualization

## Quick Start
1. Setup your environment:
```bash
sudo apt install mininet wireshark
pip install -r requirements.txt
```
2. Start Mininet topology:
```bash
cd network && sudo python3 geant_topology.py
```
3. Run training:
```bash
bash scripts/train_all.sh
```
4. Evaluate performance:
```bash
python3 scripts/evaluate.py
```
5. Plot results:
```bash
python3 scripts/plot_results.py
```
