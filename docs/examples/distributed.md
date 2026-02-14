# Distributed Systems Examples

12 examples covering consensus, replication, failure detection, and network protocols.

| Example | Description |
|---------|-------------|
| [raft_leader_election.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/raft_leader_election.py) | Raft leader election and log replication |
| [paxos_consensus.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/paxos_consensus.py) | Single-decree Paxos consensus |
| [flexible_paxos_quorums.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/flexible_paxos_quorums.py) | Flexible Paxos with asymmetric quorum configurations |
| [crdt_convergence.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/crdt_convergence.py) | CRDT eventual consistency after network partition |
| [chain_replication.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/chain_replication.py) | Chain replication with CRAQ variant |
| [primary_backup_replication.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/primary_backup_replication.py) | Primary-backup with sync/semi-sync/async modes |
| [multi_leader_replication.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/multi_leader_replication.py) | Multi-leader replication with conflict detection |
| [swim_membership.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/swim_membership.py) | SWIM membership protocol with failure detection |
| [distributed_lock_fencing.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/distributed_lock_fencing.py) | Distributed lock with fencing tokens and lease expiry |
| [tcp_congestion.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/tcp_congestion.py) | TCP congestion control: AIMD vs Cubic vs BBR |
| [dns_cache_storm.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/dns_cache_storm.py) | DNS cache miss storm under TTL expiration |
| [degraded_network.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/distributed/degraded_network.py) | Degraded network with fault injection |

## Running

```bash
python examples/distributed/raft_leader_election.py
```
