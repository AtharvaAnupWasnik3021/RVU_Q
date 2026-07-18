# SecureQ-IoT: A Trust-Aware Federated Q-Learning Routing Framework for IoT Networks

> Research-oriented implementation of a secure, privacy-aware routing framework for IoT mesh networks using **Q-Learning**, **Trust Management**, and **Federated Learning**.

---

## Overview

SecureQ-IoT is a research project developed during an internship to address secure and intelligent routing in Internet of Things (IoT) networks. The framework combines reinforcement learning with trust evaluation and federated learning to improve routing performance while protecting against malicious nodes and preserving data privacy.

Instead of relying on static routing protocols, SecureQ-IoT enables IoT nodes to learn optimal routing paths dynamically based on network conditions while continuously evaluating node trustworthiness.

---

## Objectives

- Design an adaptive routing protocol using Q-Learning.
- Detect and avoid malicious nodes through trust evaluation.
- Preserve data privacy using Federated Learning.
- Improve routing efficiency in dynamic IoT environments.
- Evaluate network performance under different attack scenarios.

---

## Features

- Intelligent Q-Learning based routing
- Trust-aware path selection
- Federated policy aggregation without sharing raw data
- Dynamic IoT network simulation
- Multiple network topologies
- Support for malicious node injection
- Performance evaluation using standard networking metrics

---

## System Architecture

```
                +----------------------+
                |   IoT Network Nodes  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  Network Simulation  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  Trust Evaluation    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   Q-Learning Agent   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Secure Route Selection|
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Federated Learning   |
                | Policy Aggregation   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Performance Analysis |
                +----------------------+
```



---

## Technologies Used

- Python 3.x
- NumPy
- NetworkX
- Matplotlib
- Reinforcement Learning (Q-Learning)
- Federated Learning
- Trust Management
- IoT Network Simulation

---

## Experimental Setup

The framework supports:

- Mesh topology
- Tree topology
- Random topology

Simulation parameters include:

- Dynamic IoT nodes
- Node mobility
- Configurable malicious node percentages
- Trust-based routing
- Federated model aggregation

---

## Performance Metrics

The framework evaluates:

- Packet Delivery Ratio (PDR)
- End-to-End Delay
- Throughput
- Energy Consumption
- Packet Loss
- Route Stability
- Trust Score
- Learning Convergence

---

## Workflow

1. Generate IoT network topology.
2. Initialize trust values for all nodes.
3. Train routing agents using Q-Learning.
4. Detect malicious nodes through trust evaluation.
5. Share learned policies via Federated Learning.
6. Select secure routing paths.
7. Evaluate network performance.

---

## Research Contributions

- Integration of Trust Management with Q-Learning for secure routing.
- Privacy-preserving routing using Federated Learning.
- Adaptive routing under dynamic network conditions.
- Modular framework for experimentation and future research.

---

## Future Improvements

- Deep Reinforcement Learning (DQN/PPO)
- Blockchain-based trust management
- Digital Twin integration
- Edge AI deployment
- Real-world IoT hardware validation
- Energy-aware routing optimization

---

## Disclaimer

This repository contains a research-oriented implementation developed as part of an internship project. It is intended for academic and experimental purposes.

---

## Author

**Atharva Anup Wasnik**

B.Tech Computer Science and Engineering

Research Interests:
- Internet of Things (IoT)
- Reinforcement Learning
- Federated Learning
- Network Security
- Artificial Intelligence

---

## License

This project is released for academic and research purposes.
