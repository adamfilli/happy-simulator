# Networking

## Network Topology

`Network` manages a graph of links between entities with configurable latency, bandwidth, and packet loss.

```python
from happysimulator import Network, datacenter_network

network = Network(name="cluster")
network.add_bidirectional_link(node_a, node_b, datacenter_network("link_ab"))
```

### Sending Events

```python
event = network.send(node_a, node_c, "Request", payload={...})
```

The event traverses the link, arriving with realistic latency.

### Network Condition Factories

| Factory | Latency | Bandwidth | Loss |
|---------|---------|-----------|------|
| `local_network()` | ~0.1ms | 1 Gbps | 0% |
| `datacenter_network()` | ~0.5ms | 10 Gbps | 0% |
| `cross_region_network()` | ~50ms | 1 Gbps | 0% |
| `internet_network()` | ~100ms | Variable | Low |
| `satellite_network()` | ~600ms | Low | Low |
| `lossy_network(rate)` | Variable | Variable | Custom |
| `slow_network()` | High | Low | 0% |
| `mobile_3g_network()` | ~100ms | 1 Mbps | 1% |
| `mobile_4g_network()` | ~30ms | 10 Mbps | 0.1% |

## Network Partitions

Simulate network failures by partitioning nodes into isolated groups:

```python
# Create a partition
partition = network.partition([node_a], [node_c], asymmetric=False)

# Events between groups are dropped
# ...

# Heal the specific partition
partition.heal()

# Or heal all partitions at once
network.heal_partition()
```

Set `asymmetric=True` for one-way partitions where messages flow in one direction but not the other.

## Next Steps

- [Clocks](clocks.md) — per-node time skew and logical clocks
- [Distributed Systems](distributed-systems.md) — consensus, replication, and coordination
