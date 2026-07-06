# Graph Chunking Plan

## Problem

Current chunking splits documents using fixed character boundaries with paragraph/sentence awareness. This can fragment coherent semantic units across unrelated topics and fails to leverage document structure (headings, references, relationships between sections).

## Solution

Build a **graph-aware chunking system** that:
1. Constructs a **text-unit graph** from documents (nodes = sentences/paras, edges = structural/semantic proximity)
2. **Partitions the graph** into communities using graph clustering (e.g., Louvain on similarity graph)
3. **Stores chunk relationships** as edges in the database
4. Uses **graph traversal** during retrieval to expand context beyond linear neighbors

---

## Architecture

```
Document
  |
  v
TextUnitExtractor_  --[sentences / paragraphs / sections]
  |
  v
GraphBuilder_       --[nodes + edges: structural + embedding similarity]
  |
  v
GraphPartitioner_   --[community detection -> GraphChunks]
  |
  v
Chunk + ChunkEdges
```

### Core Abstractions

#### `TextUnit` (in `chunker/graph.py`)
- `id`: unique unit ID
- `document_id`: parent document
- `text`: raw text
- `unit_type`: "sentence", "paragraph", "section", "heading"
- `start_char`, `end_char`: positions in source
- `metadata`: book/chapter/verse/headers

#### `ChunkGraph` (in `chunker/graph.py`)
- `nodes`: `Dict[str, TextUnit]`
- `edges`: `List[Tuple[str, str, str, float]]` (source_id, target_id, edge_type, weight)
  - Edge types: `structural` (adjacent paragraphs), `semantic` (embedding similarity), `entity` (shared named entities), `hierarchical` (parent section)

#### `GraphChunk` (extends `Chunk`)
- Same fields as `Chunk` plus:
- `unit_ids`: `List[str]` - the `TextUnit`s this chunk contains
- `graph_id`: stable identifier for graph community

---

## Data Model Changes

### Extend `chunker/chunker.py:292`
```python
@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # --- NEW ---
    unit_ids: List[str] = field(default_factory=list)
    graph_id: str = ""
    parent_chunk_id: str = ""
```

### Extend `utils/db_utils.py`

Add new table:
```sql
CREATE TABLE IF NOT EXISTS chunk_edges (
    source_chunk_id TEXT NOT NULL REFERENCES chunks(id),
    target_chunk_id TEXT NOT NULL REFERENCES chunks(id),
    edge_type TEXT DEFAULT 'semantic' CHECK (edge_type IN ('structural', 'semantic', 'entity', 'hierarchical')),
    weight FLOAT DEFAULT 1.0,
    source_document_id TEXT REFERENCES documents(id),
    PRIMARY KEY (source_chunk_id, target_chunk_id, edge_type)
);
```

Indexes:
```sql
CREATE INDEX IF NOT EXISTS chunk_edges_source_idx ON chunk_edges(source_chunk_id);
CREATE INDEX IF NOT EXISTS chunk_edges_target_idx ON chunk_edges(target_chunk_id);
CREATE INDEX IF NOT EXISTS chunk_edges_source_doc_idx ON chunk_edges(source_document_id);
```

---

## New Modules

### `chunker/graph.py` (new)

**Purpose: Build and manipulate the text-unit graph.**

```python
class TextUnit:
    id: str
    document_id: str
    text: str
    unit_type: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

class ChunkGraph:
    nodes: Dict[str, TextUnit]
    edges: List[Tuple[str, str, str, float]]  # src, dst, type, weight

    def add_node(self, unit: TextUnit) -> None: ...
    def add_edge(self, src: str, dst: str, edge_type: str, weight: float) -> None: ...
    def get_neighbors(self, unit_id: str, edge_type: Optional[str] = None) -> List[str]: ...
    def community_detection(self, resolution: float = 1.0) -> Dict[str, List[str]]: ...
    def communities_to_chunks(self, communities: Dict[str, List[str]], document_id: str) -> List["Chunk"]: ...
```

**Key functions:**
- `build_graph_from_document(doc: Document, chunker: Chunker) -> ChunkGraph` - extract text units and create structural/semantic/hierarchical edges
- `build_semantic_edges(graph: ChunkGraph, model, threshold: float = 0.7) -> ChunkGraph` - use embedding similarity to add semantic edges
- `build_entity_edges(graph: ChunkGraph) -> ChunkGraph` - use keyword co-occurrence as entity proxy
- `partition_graph(graph: ChunkGraph, min_community_size: int = 2) -> Dict[str, list]` - Louvain community detection
- `validate_communities(graph: ChunkGraph, communities: Dict[str, list], document_id: str) -> List[ChunkFix]` - ensure no orphaned tiny communities

### `chunker/graph_store.py` (new)

**Purpose: Persist and query chunk graph relationships.**

```python
class GraphStore:
    def add_chunk(self, chunk: Chunk) -> None: ...
    def add_edge(self, source_id: str, target_id: str, edge_type: str, weight: float, doc_id: str) -> None: ...
    def get_neighbors(self, chunk_id: str, edge_types: Optional[List[str]] = None, top_k: int = 10) -> List[Tuple[str, str, float]]: ...
    def get_ego_graph(self, chunk_id: str, hops: int = 1, edge_types: Optional[List[str]] = None) -> Set[str]: ...
    def build_from_file(self, chunk_file: str) -> None: ...
    def clear(self) -> None: ...
```

### `chunker/__init__.py` updates

Export new types and classes:
```python
from chunker.graph import ChunkGraph, TextUnit, GraphStore
```

---

## Existing Module Changes

### `chunker/chunker.py`
- Add `graph_mode` parameter to `Chunker.__init__` (default `False` for backward compat)
- Add `create_graph_chunks(self, document_id, content, metadata=None, model=None) -> List[Chunk]`
- Modify `create_chunks` to optionally accept `graph=True`
- When `graph_mode=False`, behavior is identical to current implementation

### `chunker/document.py`
- No breaking changes. `Chunk.from_dict()` should handle new fields (`unit_ids`, `graph_id`, `parent_chunk_id`) gracefully

### `utils/db_utils.py`
- Add `SQL_CREATE_CHUNK_EDGES_TABLE` and DDL constants
- Add `insert_chunk_edge(cur, ...)`, `get_chunk_neighbors(cur, chunk_id, edge_types, top_k)`
- Add `get_ego_chunks(cur, chunk_id, hops)` for multi-hop expansion
- Update `create_tables()` to also create edges table and indexes

### `vector_store/db_store.py`
- Add `save_graph_edges(chunk_edges)` after embedding build
- Add `get_graph_neighbors(chunk_id, edge_types, top_k)` method

### `processor/processor.py`
- Update `_expand_candidate_chunks` (line 129) to also consider graph neighbors
- New function: `_graph_expand_candidates(store, candidate_ids, hops=2) -> Set[str]`
- Hybrid expansion: linear adjacency (±3 chunks) + graph neighbors (semantic/hierarchical)
- In `_rank_results`, add graph-based score adjustment (PageRank-like influence)

### `app.py`
- Add `--graph/--no-graph` flag to `scrape`, `pdf_scan`, `build_chunk_cache` commands
- Graph mode triggers `chunker.create_graph_chunks()` and `store.save_graph_edges()`

---

## Graph Construction Details

### 1. Structural Edges
- Adjacent sentences in same paragraph
- Adjacent paragraphs within a chunk
- Section -> subsection encapsulation
- Book -> chapter -> verse hierarchy (for structured docs like PDF books)

### 2. Semantic Edges
- Compute sentence/paragraph embeddings via existing `embed_texts()`
- Connect units with cosine similarity > threshold (e.g., 0.65)
- Limit degree to `max(5, sqrt(unit_count))` to prevent dense graphs

### 3. Entity/Hierarchical Edges
- Detect shared keywords or heading paths
- Connect topics discussed in same section/heading
- Track parent-child relationships in document outline

---

## Algorithm Selection

### Community Detection
**Recommendation: Louvain method (greedy modularity)**  
- O(n log n) time complexity with optimized networkx-like approach
- Minimal additional dependencies (can use `community-louvain` package or implement a simple greedy version)
- Produces hierarchical communities (can stop at desired chunk resolution)

**Fallback**: Greedy connected component merge based on semantic similarity threshold.

### Graph Traversal for Retrieval
- **1-hop**: Immediate neighbors
- **2-hop**: Neighbors of neighbors (breadth-limited to avoid explosion)
- Score propagation: multiply neighbor scores by edge weight, decay by 0.5 per hop

---

## Dependencies

### New (optional but recommended)
```
networkx>=3.0
python-louvain>=0.16   # for community detection
# OR custom lightweight greedy community detection (no extra dep)
```

### Alternative (zero new deps)
- Use `sklearn.cluster.AgglomerativeClustering` or `DBSCAN` on embeddings to form communities
- This avoids graph construction entirely but loses structural edges

---

## Implementation Phases

### Phase 1: Structural Graph Chunker (foundation, ~2 days)
1. Create `chunker/graph.py` with `TextUnit`, `ChunkGraph`
2. Implement `extract_text_units()`, `build_structural_edges()`
3. Implement simple community detection (connected component merge by similarity)
4. Add `create_graph_chunks()` to `Chunker`
5. Basic tests for graph building and chunk output

### Phase 2: Graph Storage and Schema (database, ~1 day)
1. Add `chunk_edges` table to `utils/db_utils.py`
2. Add `GraphStore` class in `chunker/graph_store.py`
3. Update `vector_store/db_store.py` to persist edges alongside chunks
4. Update `chunker/graph.py` to output graph alongside chunks

### Phase 3: Graph-Enhanced Retrieval (RAG, ~2 days)
1. Update `processor/processor.py` `_expand_candidate_chunks`
2. New `_graph_expand_candidates()` using stored edges
3. Multi-hop BFS expansion during search
4. A/B baseline: test retrieval quality (Recall@k) vs current linear expansion

### Phase 4: Semantic and Entity Edges (enhancement, ~1-2 days)
1. Add `build_semantic_edges()` using sentence embeddings
2. Add `build_entity_edges()` using keyword co-occurrence
3. Replace simple community detection with full Louvain
4. Tune edge weights and community resolution

---

## File Inventory

| File | Action | Description |
|------|--------|-------------|
| `chunker/graph.py` | **New** | Core graph data structures and algorithms |
| `chunker/graph_store.py` | **New** | Persistence layer for graph edges |
| `chunker/graph_index.py` | **New** | In-memory index for graph traversal in file backend |
| `chunker/chunker.py` | **Modify** | Add `graph_mode`, `create_graph_chunks()` |
| `chunker/document.py` | **Modify** | Handle new `Chunk` fields in `from_dict()` |
| `chunker/__init__.py` | **Modify** | Export new classes |
| `utils/db_utils.py` | **Modify** | Add edges DDL, queries, helpers |
| `vector_store/db_store.py` | **Modify** | Edge persistence, neighbor lookup |
| `vector_store/index.py` | **Modify** | Graph index for file backend |
| `processor/processor.py` | **Modify** | Graph-aware candidate expansion |
| `app.py` | **Modify** | Graph mode CLI flags |
| `config/config.cfg.example` | **Modify** | New `[graph]` section |

---

## Config Section

```ini
[graph]
enabled = false
community_resolution = 1.0
semantic_threshold = 0.65
similarity_model = paraphrase-multilingual-MiniLM-L12-v2
max_neighbor_degree = 10
expansion_hops = 2
expansion_decay = 0.5
```

---

## Open Questions

1. **Zero-dependency vs optimal accuracy**: Should we add `networkx`/`python-louvain`, or implement a lightweight greedy algorithm to keep dependencies minimal?
2. **Graph chunk size control**: Current `chunk_size` is characters. Graph chunks could be targeting target community size (e.g., 500-1000 chars). Should we parameterize community detection by target chunk count or let it flow naturally?
3. **Mixed mode**: Should graph chunks coexist with standard chunks, or replace them?
4. **Performance**: Semantic edge computation is O(n^2) embeddings. For large docs, need minibatch or approximate nearest neighbors (FAISS). For now, cap semantic edges to chunks with section overlap or random sampling.
