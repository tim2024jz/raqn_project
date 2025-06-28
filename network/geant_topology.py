#!/usr/bin/python3
"""
GEANT-like Mininet topology generator
- Creates a topology with N switches and L random links
- Each link has a random delay in [1, 200] ms
- Designed for SDN experiments with external controller (e.g., Ryu)
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import RemoteController
from mininet.cli import CLI
import random
import sys

class GEANTTopo(Topo):
    def build(self, num_nodes=22, num_links=72):
        if num_links > num_nodes * (num_nodes - 1) // 2:
            sys.exit(f"[ERROR] Requested {num_links} links exceeds maximum unique links "
                     f"possible between {num_nodes} nodes.")

        switches = []
        for i in range(num_nodes):
            switch = self.addSwitch(f's{i+1}')
            switches.append(switch)

        connected = set()
        attempts = 0
        max_attempts = num_links * 10  # Avoid infinite loops in degenerate cases

        while len(connected) < num_links and attempts < max_attempts:
            a, b = random.sample(range(num_nodes), 2)
            if (a, b) not in connected and (b, a) not in connected:
                delay = f"{random.randint(1,200)}ms"
                bw = random.choice([10, 20, 50, 100])  # bandwidth options in Mbps
                loss = random.uniform(0, 2)  # up to 2% loss
                self.addLink(
                    switches[a], switches[b],
                    cls=TCLink,
                    delay=delay,
                    bw=bw,
                    loss=round(loss, 2)
                )
                connected.add((a, b))
            attempts += 1

        if len(connected) < num_links:
            sys.exit(f"[ERROR] Could only create {len(connected)} links after {attempts} attempts. "
                     f"Consider reducing num_links or increasing max_attempts.")

if __name__ == '__main__':
    random.seed(42)  # deterministic randomness for repeatable experiments
    num_nodes = 22
    num_links = 72
    controller_ip = '127.0.0.1'
    controller_port = 6633

    topo = GEANTTopo(num_nodes=num_nodes, num_links=num_links)
    net = Mininet(topo=topo, controller=None, autoSetMacs=True, link=TCLink)

    c0 = net.addController('c0', controller=RemoteController, ip=controller_ip, port=controller_port)

    print(f"*** Starting Mininet with {num_nodes} nodes and {num_links} links.")
    print(f"*** Controller: {controller_ip}:{controller_port}")
    net.start()
    print("*** Network started with GEANT-like topology.")
    CLI(net)
    net.stop()
    print("*** Network stopped.")
