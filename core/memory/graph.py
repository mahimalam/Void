"""Entity-Relation Knowledge Graph Engine.

Provides an Associative Memory Network (Hebbian Graph) over your memories.
Uses SQLite for persistent storage of triples (subject, predicate, object)
and NetworkX for extremely fast, in-memory Spreading Activation.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import networkx as nx

from core.logging_setup import get_logger
from core.db import _ensure_dir

_log = get_logger("memory_graph")

GRAPH_DDL = """
CREATE TABLE IF NOT EXISTS graph_triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    is_pinned BOOLEAN DEFAULT 0,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject, predicate, object)
)
"""

class KnowledgeGraph:
    """Associative Memory Network for fast, bidirectional entity traversal."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        _ensure_dir(self.db_path)
        self._graph = nx.MultiDiGraph()
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize SQLite storage and load the graph into memory."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA busy_timeout = 5000;")
                await conn.execute(GRAPH_DDL)
                # Migration: ensure is_pinned column exists for existing databases
                try:
                    await conn.execute("ALTER TABLE graph_triples ADD COLUMN is_pinned BOOLEAN DEFAULT 0;")
                except Exception:
                    pass  # Column already exists
                await conn.commit()
            
            await self._load_into_memory()
            _log.info("knowledge_graph_initialized", nodes=self._graph.number_of_nodes(), edges=self._graph.number_of_edges())
        except Exception as e:
            _log.error("knowledge_graph_init_failed", error=str(e))
            raise

    async def _load_into_memory(self) -> None:
        """Load all triples from SQLite into the NetworkX MultiDiGraph."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute("PRAGMA busy_timeout = 5000;")
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute("SELECT subject, predicate, object, weight, is_pinned FROM graph_triples")
                rows = await cursor.fetchall()
                
                async with self._lock:
                    self._graph.clear()
                    for row in rows:
                        self._graph.add_edge(
                            row["subject"],
                            row["object"],
                            key=row["predicate"],
                            predicate=row["predicate"],
                            weight=row["weight"],
                            is_pinned=bool(row["is_pinned"] if "is_pinned" in row.keys() else False)
                        )
        except Exception as e:
            _log.error("knowledge_graph_load_failed", error=str(e))

    async def add_edge(
        self, subject: str, predicate: str, obj: str, weight: float = 1.0, is_pinned: bool = False
    ) -> None:
        """Add or reinforce a directed edge. Pinned edges are permanently immune to decay."""
        subject = subject.strip().lower()
        predicate = predicate.strip().lower()
        obj = obj.strip().lower()

        if not subject or not predicate or not obj:
            return

        try:
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute("PRAGMA busy_timeout = 5000;")
                pinned_val = 1 if is_pinned else 0
                await conn.execute("""
                    INSERT INTO graph_triples (subject, predicate, object, weight, is_pinned, last_accessed)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(subject, predicate, object) DO UPDATE SET
                        weight = weight + ?,
                        is_pinned = MAX(is_pinned, ?),
                        last_accessed = CURRENT_TIMESTAMP
                """, (subject, predicate, obj, weight, pinned_val, weight, pinned_val))
                await conn.commit()

            async with self._lock:
                if self._graph.has_edge(subject, obj, key=predicate):
                    current_weight = self._graph[subject][obj][predicate].get("weight", 1.0)
                    self._graph[subject][obj][predicate]["weight"] = current_weight + weight
                    if is_pinned:
                        self._graph[subject][obj][predicate]["is_pinned"] = True
                else:
                    self._graph.add_edge(
                        subject, obj, key=predicate, predicate=predicate, weight=weight, is_pinned=is_pinned
                    )
                    
            _log.debug("knowledge_graph_edge_added", subject=subject, predicate=predicate, obj=obj, is_pinned=is_pinned)
        except Exception as e:
            _log.error("knowledge_graph_add_failed", error=str(e))

    async def spread_activation(
        self, start_nodes: list[str], max_hops: int = 2, threshold: float = 0.3
    ) -> list[dict[str, Any]]:
        """Perform bidirectional associative Spreading Activation with hyperbolic tangent bounding."""
        if not start_nodes:
            return []
            
        async with self._lock:
            start_nodes = [n.strip().lower() for n in start_nodes if self._graph.has_node(n.strip().lower())]
            if not start_nodes:
                return []
                
            activations: dict[str, float] = {n: 1.0 for n in start_nodes}
            
            # BFS with decay and bidirectional spreading
            for _ in range(max_hops):
                next_activations: dict[str, float] = {}
                for node, act in activations.items():
                    if act < threshold:
                        continue
                    
                    # 1. Forward traversal: outgoing edges (node -> neighbor)
                    for neighbor in self._graph.successors(node):
                        edge_dict = self._graph[node][neighbor]
                        max_w = max(data.get("weight", 1.0) for data in edge_dict.values())
                        bounded_weight = math.tanh(max_w)
                        spread_amount = min(1.0, act * (bounded_weight * 0.8))
                        
                        next_activations[neighbor] = max(next_activations.get(neighbor, 0.0), spread_amount)

                    # 2. Reverse traversal: incoming edges (neighbor -> node)
                    # Human memory connects both ways: 'tesla' -> 'vexp' just as 'vexp' -> 'tesla'
                    for neighbor in self._graph.predecessors(node):
                        edge_dict = self._graph[neighbor][node]
                        max_w = max(data.get("weight", 1.0) for data in edge_dict.values())
                        bounded_weight = math.tanh(max_w)
                        spread_amount = min(1.0, act * (bounded_weight * 0.7))  # 0.7 slight reverse decay
                        
                        next_activations[neighbor] = max(next_activations.get(neighbor, 0.0), spread_amount)
                
                # Merge next_activations into activations
                for n, act in next_activations.items():
                    activations[n] = max(activations.get(n, 0.0), act)
                        
            # Filter and sort results
            results = []
            for node, activation in sorted(activations.items(), key=lambda x: x[1], reverse=True):
                if activation >= threshold:
                    connections = []
                    # Incoming connections
                    for pred in self._graph.predecessors(node):
                        if pred in activations:
                            for key, edge_data in self._graph[pred][node].items():
                                rel = edge_data.get("predicate", "related_to")
                                connections.append(f"{pred} -[{rel}]-> {node}")
                    # Outgoing connections
                    for succ in self._graph.successors(node):
                        if succ in activations:
                            for key, edge_data in self._graph[node][succ].items():
                                rel = edge_data.get("predicate", "related_to")
                                connections.append(f"{node} -[{rel}]-> {succ}")
                                
                    results.append({
                        "node": node,
                        "activation": min(1.0, round(activation, 3)),
                        "connections": list(set(connections))[:5]  # Deduplicate & cap to 5
                    })
                    
            return results

    async def apply_synaptic_decay(self, decay_rate: float = 0.05, prune_threshold: float = 0.1) -> int:
        """The Midnight Sweeper: Decay all non-pinned weights and prune dead links."""
        pruned_count = 0
        try:
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute("PRAGMA busy_timeout = 5000;")
                # Decay only non-pinned triples
                await conn.execute("UPDATE graph_triples SET weight = weight * (1.0 - ?) WHERE is_pinned = 0", (decay_rate,))
                
                # Prune only non-pinned dead links
                cursor = await conn.execute("SELECT count(*) FROM graph_triples WHERE weight < ? AND is_pinned = 0", (prune_threshold,))
                row = await cursor.fetchone()
                pruned_count = row[0] if row else 0
                
                await conn.execute("DELETE FROM graph_triples WHERE weight < ? AND is_pinned = 0", (prune_threshold,))
                await conn.commit()
                
            # Reload graph to reflect pruned state
            await self._load_into_memory()
            _log.info("knowledge_graph_sweeper_ran", decayed_by=decay_rate, pruned_edges=pruned_count)
            return pruned_count
        except Exception as e:
            _log.error("knowledge_graph_sweeper_failed", error=str(e))
            return 0
