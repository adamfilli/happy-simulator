# Distributed Systems

happy-simulator includes full implementations of distributed systems algorithms for simulation and educational use.

## Consensus

### Raft

```python
from happysimulator import RaftNode, Network, datacenter_network

nodes = [RaftNode(f"node-{i}") for i in range(5)]
network = Network(name="cluster")
# Add links between all nodes...

# Raft handles leader election, log replication, and membership changes
```

See `examples/distributed/raft_leader_election.py` for a complete example.

### Paxos

```python
from happysimulator import PaxosNode, MultiPaxosNode, FlexiblePaxosNode
```

- `PaxosNode` — single-decree Paxos
- `MultiPaxosNode` — multi-decree with stable leader
- `FlexiblePaxosNode` — configurable quorum sizes

### Leader Election

```python
from happysimulator import LeaderElection, BullyStrategy, RingStrategy, RandomizedStrategy

election = LeaderElection(
    name="election",
    strategy=BullyStrategy(),
    network=network,
)
```

## Distributed Locks

```python
from happysimulator import DistributedLock

lock = DistributedLock(name="mutex", nodes=nodes, network=network)
# Acquire returns a LockGrant with fencing token
grant = yield lock.acquire()
# ... critical section ...
grant.release()
```

## Failure Detection

```python
from happysimulator import PhiAccrualDetector

detector = PhiAccrualDetector()
detector.heartbeat(timestamp)
phi = detector.phi(now)  # suspicion level (>8 usually means dead)
```

## Membership

```python
from happysimulator import MembershipProtocol

protocol = MembershipProtocol(name="swim", nodes=nodes, network=network)
# SWIM protocol for scalable failure detection and membership
```

## CRDTs

Conflict-free Replicated Data Types for eventual consistency:

```python
from happysimulator import GCounter, PNCounter, LWWRegister, ORSet, CRDTStore

counter = GCounter("node-1")
counter.increment(5)
counter.merge(remote_counter.state())

store = CRDTStore(name="store", node_id="node-1", network=network)
```

## Replication

- Primary-backup replication
- Multi-leader replication
- Chain replication

See `examples/distributed/` for complete examples of each pattern.

## Next Steps

- [Networking](networking.md) — network topology and partitions
- [Clocks](clocks.md) — time skew and logical ordering
- [Fault Injection](fault-injection.md) — systematic failure testing
