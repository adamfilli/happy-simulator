"""Election strategies for leader election protocols.

Provides pluggable election algorithms: Bully (highest ID wins),
Ring (token circulation), and Randomized (random ballot numbers).
"""

from __future__ import annotations

import random
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ElectionStrategy(Protocol):
    """Protocol for pluggable election algorithms."""

    def should_start_election(self, node_id: str, alive_members: list[str]) -> bool:
        """Determine if this node should initiate an election.

        Args:
            node_id: This node's identifier.
            alive_members: List of alive member identifiers.

        Returns:
            True if this node should start an election.
        """
        ...

    def get_election_messages(
        self, node_id: str, alive_members: list[str], term: int
    ) -> list[dict[str, Any]]:
        """Generate election messages to send to peers.

        Args:
            node_id: This node's identifier.
            alive_members: List of alive member identifiers.
            term: Current election term.

        Returns:
            List of message dicts with 'target', 'event_type', 'payload'.
        """
        ...

    def handle_election_message(
        self, node_id: str, message_type: str, payload: dict[str, Any],
        alive_members: list[str],
    ) -> dict[str, Any]:
        """Process an incoming election message.

        Args:
            node_id: This node's identifier.
            message_type: The type of election message.
            payload: Message payload.
            alive_members: List of alive member identifiers.

        Returns:
            Dict with 'response_messages' (list), 'leader' (str|None),
            'suppress_election' (bool).
        """
        ...


class BullyStrategy:
    """Bully election: highest ID wins.

    When a node detects the leader is down, it sends Election messages to
    all nodes with higher IDs. If none respond, it declares itself leader.
    Nodes with higher IDs override lower ones by sending Bully messages.
    """

    def should_start_election(self, node_id: str, alive_members: list[str]) -> bool:
        return True

    def get_election_messages(
        self, node_id: str, alive_members: list[str], term: int,
    ) -> list[dict[str, Any]]:
        higher = [m for m in alive_members if m > node_id]
        if not higher:
            # No higher nodes → declare victory
            return [
                {
                    "target": m,
                    "event_type": "ElectionVictory",
                    "payload": {"leader": node_id, "term": term},
                }
                for m in alive_members
                if m != node_id
            ]
        # Send Election to higher-ID nodes
        return [
            {
                "target": m,
                "event_type": "ElectionChallenge",
                "payload": {"challenger": node_id, "term": term},
            }
            for m in higher
        ]

    def handle_election_message(
        self, node_id: str, message_type: str, payload: dict[str, Any],
        alive_members: list[str],
    ) -> dict[str, Any]:
        if message_type == "ElectionChallenge":
            challenger = payload.get("challenger", "")
            if node_id > challenger:
                # We override the challenger
                return {
                    "response_messages": [
                        {
                            "target": challenger,
                            "event_type": "ElectionSuppress",
                            "payload": {"from": node_id},
                        }
                    ],
                    "leader": None,
                    "suppress_election": False,
                    "start_own_election": True,
                }
            return {"response_messages": [], "leader": None, "suppress_election": False}

        if message_type == "ElectionSuppress":
            return {"response_messages": [], "leader": None, "suppress_election": True}

        if message_type == "ElectionVictory":
            leader = payload.get("leader")
            return {"response_messages": [], "leader": leader, "suppress_election": True}

        return {"response_messages": [], "leader": None, "suppress_election": False}


class RingStrategy:
    """Ring election: token circulates, highest ID in the ring wins.

    The election token travels around a logical ring. Each node appends
    its ID. When the token returns to the initiator, the highest ID wins.
    """

    def should_start_election(self, node_id: str, alive_members: list[str]) -> bool:
        return True

    def get_election_messages(
        self, node_id: str, alive_members: list[str], term: int,
    ) -> list[dict[str, Any]]:
        # Start ring token with self
        ring = sorted([m for m in alive_members if m != node_id] + [node_id])
        idx = ring.index(node_id)
        next_node = ring[(idx + 1) % len(ring)]
        return [
            {
                "target": next_node,
                "event_type": "ElectionToken",
                "payload": {
                    "initiator": node_id,
                    "candidates": [node_id],
                    "term": term,
                },
            }
        ]

    def handle_election_message(
        self, node_id: str, message_type: str, payload: dict[str, Any],
        alive_members: list[str],
    ) -> dict[str, Any]:
        if message_type == "ElectionToken":
            initiator = payload["initiator"]
            candidates = list(payload["candidates"])

            if initiator == node_id:
                # Token came back — pick winner
                leader = max(candidates)
                return {
                    "response_messages": [
                        {
                            "target": m,
                            "event_type": "ElectionVictory",
                            "payload": {"leader": leader, "term": payload.get("term", 0)},
                        }
                        for m in alive_members
                        if m != node_id
                    ],
                    "leader": leader,
                    "suppress_election": True,
                }

            # Add self and forward
            candidates.append(node_id)
            ring = sorted([m for m in alive_members if m != node_id] + [node_id])
            idx = ring.index(node_id)
            next_node = ring[(idx + 1) % len(ring)]
            return {
                "response_messages": [
                    {
                        "target": next_node,
                        "event_type": "ElectionToken",
                        "payload": {
                            "initiator": initiator,
                            "candidates": candidates,
                            "term": payload.get("term", 0),
                        },
                    }
                ],
                "leader": None,
                "suppress_election": False,
            }

        if message_type == "ElectionVictory":
            return {
                "response_messages": [],
                "leader": payload.get("leader"),
                "suppress_election": True,
            }

        return {"response_messages": [], "leader": None, "suppress_election": False}


class RandomizedStrategy:
    """Randomized election: random ballot numbers, highest wins.

    Each participant draws a random ballot. After collecting ballots
    from all alive members, the highest ballot wins.
    """

    def __init__(self, ballot_range: int = 1_000_000) -> None:
        self._ballot_range = ballot_range

    def should_start_election(self, node_id: str, alive_members: list[str]) -> bool:
        return True

    def get_election_messages(
        self, node_id: str, alive_members: list[str], term: int,
    ) -> list[dict[str, Any]]:
        ballot = random.randint(1, self._ballot_range)
        return [
            {
                "target": m,
                "event_type": "ElectionBallot",
                "payload": {
                    "from": node_id,
                    "ballot": ballot,
                    "term": term,
                },
            }
            for m in alive_members
            if m != node_id
        ]

    def handle_election_message(
        self, node_id: str, message_type: str, payload: dict[str, Any],
        alive_members: list[str],
    ) -> dict[str, Any]:
        if message_type == "ElectionBallot":
            # Respond with own ballot
            my_ballot = random.randint(1, self._ballot_range)
            sender = payload.get("from")
            return {
                "response_messages": [
                    {
                        "target": sender,
                        "event_type": "ElectionBallotResponse",
                        "payload": {
                            "from": node_id,
                            "ballot": my_ballot,
                            "term": payload.get("term", 0),
                        },
                    }
                ] if sender else [],
                "leader": None,
                "suppress_election": False,
            }

        if message_type == "ElectionVictory":
            return {
                "response_messages": [],
                "leader": payload.get("leader"),
                "suppress_election": True,
            }

        return {"response_messages": [], "leader": None, "suppress_election": False}
