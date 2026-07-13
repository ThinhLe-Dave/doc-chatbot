import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, sys.path[0] + "/..")


class TextUnitTest(unittest.TestCase):
    def test_roundtrip_serialization(self):
        from chunker.graph import TextUnit

        unit = TextUnit(
            unit_id="u1",
            document_id="d1",
            text="Hello world",
            unit_type="sentence",
            metadata={"page": 1},
            index=0,
        )
        data = unit.to_dict()
        restored = TextUnit.from_dict(data)
        self.assertEqual(restored.unit_id, "u1")
        self.assertEqual(restored.document_id, "d1")
        self.assertEqual(restored.text, "Hello world")
        self.assertEqual(restored.unit_type, "sentence")
        self.assertEqual(restored.metadata, {"page": 1})
        self.assertEqual(restored.index, 0)

    def test_defaults_on_from_dict(self):
        from chunker.graph import TextUnit

        unit = TextUnit.from_dict({})
        self.assertEqual(unit.unit_id, "")
        self.assertEqual(unit.document_id, "")
        self.assertEqual(unit.text, "")
        self.assertEqual(unit.unit_type, "sentence")
        self.assertEqual(unit.metadata, {})
        self.assertEqual(unit.index, 0)

    def test_content_hash_is_deterministic(self):
        from chunker.graph import TextUnit

        unit = TextUnit(unit_id="u1", document_id="d1", text="hello", index=0)
        first = unit.content_hash()
        second = unit.content_hash()
        self.assertEqual(first, second)
        self.assertTrue(len(first) == 64)


class GraphEdgeTest(unittest.TestCase):
    def test_roundtrip_serialization(self):
        from chunker.graph import GraphEdge

        edge = GraphEdge(source="s1", target="t1", edge_type="semantic", weight=0.9)
        data = edge.to_dict()
        restored = GraphEdge.from_dict(data)
        self.assertEqual(restored.source, "s1")
        self.assertEqual(restored.target, "t1")
        self.assertEqual(restored.edge_type, "semantic")
        self.assertAlmostEqual(restored.weight, 0.9)

    def test_defaults(self):
        from chunker.graph import GraphEdge

        edge = GraphEdge.from_dict({})
        self.assertEqual(edge.source, "")
        self.assertEqual(edge.target, "")
        self.assertEqual(edge.edge_type, "structural")
        self.assertAlmostEqual(edge.weight, 1.0)


class ChunkGraphStructuralEdgesTest(unittest.TestCase):
    def test_structural_edges_are_bidirectional(self):
        from chunker.graph import ChunkGraph, TextUnit

        graph = ChunkGraph()
        units = [
            TextUnit(unit_id="u0", document_id="d1", text="a", index=0),
            TextUnit(unit_id="u1", document_id="d1", text="b", index=1),
            TextUnit(unit_id="u2", document_id="d1", text="c", index=2),
        ]
        for unit in units:
            graph.add_unit(unit)
        graph.build_structural_edges(units)

        self.assertEqual(len(graph.edges), 4)
        types = {e.edge_type for e in graph.edges}
        self.assertEqual(types, {"structural"})

    def test_empty_units_produce_no_edges(self):
        from chunker.graph import ChunkGraph

        graph = ChunkGraph()
        graph.build_structural_edges([])
        self.assertEqual(len(graph.edges), 0)


class ChunkGraphHierarchicalEdgesTest(unittest.TestCase):
    def test_hierarchical_edges_grouped_by_chapter(self):
        from chunker.graph import ChunkGraph, TextUnit

        graph = ChunkGraph()
        units = [
            TextUnit(unit_id="u0", document_id="d1", text="a", index=0, metadata={"chapter": "1"}),
            TextUnit(unit_id="u1", document_id="d1", text="b", index=1, metadata={"chapter": "1"}),
            TextUnit(unit_id="u2", document_id="d1", text="c", index=2, metadata={"chapter": "2"}),
        ]
        for unit in units:
            graph.add_unit(unit)
        graph.build_hierarchical_edges(units)

        pairs = {(e.source, e.target) for e in graph.edges}
        self.assertIn(("u0", "u1"), pairs)
        self.assertNotIn(("u1", "u2"), pairs)
        for e in graph.edges:
            self.assertEqual(e.edge_type, "hierarchical")
            self.assertAlmostEqual(e.weight, 0.5)


class ChunkGraphSemanticEdgesTest(unittest.TestCase):
    def test_semantic_edges_below_threshold_are_skipped(self):
        from chunker.graph import ChunkGraph, TextUnit

        graph = ChunkGraph()
        units = [
            TextUnit(unit_id="u0", document_id="d1", text="a", index=0, metadata={"embedding": [1.0, 0.0]}),
            TextUnit(unit_id="u1", document_id="d1", text="b", index=1, metadata={"embedding": [0.0, 1.0]}),
        ]
        for unit in units:
            graph.add_unit(unit)
        graph.build_semantic_edges(units, threshold=0.99)
        self.assertEqual(len(graph.edges), 0)

    def test_semantic_edges_above_threshold_are_added(self):
        from chunker.graph import ChunkGraph, TextUnit

        graph = ChunkGraph()
        units = [
            TextUnit(unit_id="u0", document_id="d1", text="a", index=0, metadata={"embedding": [1.0, 0.0]}),
            TextUnit(unit_id="u1", document_id="d1", text="b", index=1, metadata={"embedding": [1.0, 0.0]}),
        ]
        for unit in units:
            graph.add_unit(unit)
        graph.build_semantic_edges(units, threshold=0.5)
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].edge_type, "semantic")


class ChunkGraphCommunityDetectionTest(unittest.TestCase):
    def test_isolated_nodes_each_get_own_community(self):
        from chunker.graph import ChunkGraph, TextUnit

        graph = ChunkGraph()
        units = [
            TextUnit(unit_id="u0", document_id="d1", text="a", index=0),
            TextUnit(unit_id="u1", document_id="d1", text="b", index=1),
        ]
        for unit in units:
            graph.add_unit(unit)
        comms = graph.detect_communities()
        self.assertEqual(len(set(comms.values())), 2)

    def test_strongly_connected_nodes_merge(self):
        from chunker.graph import ChunkGraph, TextUnit, GraphEdge

        graph = ChunkGraph()
        units = [
            TextUnit(unit_id="u0", document_id="d1", text="a", index=0),
            TextUnit(unit_id="u1", document_id="d1", text="b", index=1),
        ]
        for unit in units:
            graph.add_unit(unit)
        graph.add_edge(GraphEdge(source="u0", target="u1", edge_type="structural", weight=10.0))
        graph.add_edge(GraphEdge(source="u1", target="u0", edge_type="structural", weight=10.0))
        comms = graph.detect_communities(resolution=0.0)
        self.assertEqual(len(set(comms.values())), 1)


class ChunkGraphNeighborsTest(unittest.TestCase):
    def test_neighbors_are_undirected_and_bounded(self):
        from chunker.graph import ChunkGraph, TextUnit, GraphEdge

        graph = ChunkGraph()
        units = [
            TextUnit(unit_id="u0", document_id="d1", text="a", index=0),
            TextUnit(unit_id="u1", document_id="d1", text="b", index=1),
            TextUnit(unit_id="u2", document_id="d1", text="c", index=2),
        ]
        for unit in units:
            graph.add_unit(unit)
        graph.add_edge(GraphEdge(source="u0", target="u1", edge_type="structural"))
        graph.add_edge(GraphEdge(source="u1", target="u2", edge_type="structural"))

        n0 = graph.neighbors("u0", max_hops=1)
        self.assertIn("u1", n0)
        self.assertNotIn("u2", n0)

        n0_2 = graph.neighbors("u0", max_hops=2)
        self.assertIn("u1", n0_2)
        self.assertIn("u2", n0_2)

    def test_neighbors_empty_when_isolated(self):
        from chunker.graph import ChunkGraph, TextUnit

        graph = ChunkGraph()
        graph.add_unit(TextUnit(unit_id="u0", document_id="d1", text="a", index=0))
        self.assertEqual(graph.neighbors("u0"), [])


class ChunkGraphSerializationTest(unittest.TestCase):
    def test_roundtrip(self):
        from chunker.graph import ChunkGraph, TextUnit, GraphEdge

        graph = ChunkGraph()
        units = [
            TextUnit(unit_id="u0", document_id="d1", text="a", index=0),
            TextUnit(unit_id="u1", document_id="d1", text="b", index=1),
        ]
        for unit in units:
            graph.add_unit(unit)
        graph.add_edge(GraphEdge(source="u0", target="u1", edge_type="structural"))

        data = graph.to_dict()
        restored = ChunkGraph.from_dict(data)
        self.assertEqual(len(restored.units), 2)
        self.assertEqual(len(restored.edges), 1)
        self.assertIn("u0", restored.units)
        self.assertEqual(restored.edges[0].source, "u0")


class ChunkBackwardCompatTest(unittest.TestCase):
    def test_old_dict_without_graph_fields(self):
        from chunker.chunker import Chunk

        old = {
            "id": "c1",
            "document_id": "d1",
            "content": "hello",
            "path": ["/a"],
            "metadata": {"title": "T"},
        }
        chunk = Chunk.from_dict(old)
        self.assertEqual(chunk.unit_ids, [])
        self.assertIsNone(chunk.graph_id)
        self.assertIsNone(chunk.parent_chunk_id)
        data = chunk.to_dict()
        self.assertIn("unit_ids", data)
        self.assertIn("graph_id", data)
        self.assertIn("parent_chunk_id", data)

    def test_graph_fields_roundtrip(self):
        from chunker.chunker import Chunk

        chunk = Chunk(
            id="c1",
            document_id="d1",
            content="hello",
            path=["/a"],
            metadata={"title": "T"},
            unit_ids=["u1", "u2"],
            graph_id="g1",
            parent_chunk_id="p1",
        )
        restored = Chunk.from_dict(chunk.to_dict())
        self.assertEqual(restored.unit_ids, ["u1", "u2"])
        self.assertEqual(restored.graph_id, "g1")
        self.assertEqual(restored.parent_chunk_id, "p1")


class ChunkerGraphModeTest(unittest.TestCase):
    def test_create_graph_chunks_returns_graph(self):
        from chunker.chunker import Chunker
        from chunker.document import Document

        doc = Document.create(
            source="https://example.com",
            title="Test",
            content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        )
        chunker = Chunker()
        chunks, graph = chunker.create_graph_chunks(doc)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(len(graph.units), 3)
        self.assertGreater(len(graph.edges), 0)

    def test_graph_chunks_have_graph_metadata(self):
        from chunker.chunker import Chunker
        from chunker.document import Document

        doc = Document.create(
            source="https://example.com",
            title="Test",
            content="Para one.\n\nPara two.",
        )
        chunker = Chunker()
        chunks, _ = chunker.create_graph_chunks(doc)
        for chunk in chunks:
            self.assertIsInstance(chunk.unit_ids, list)
            self.assertTrue(len(chunk.unit_ids) > 0)
            self.assertTrue(chunk.graph_id.startswith(doc.id + "_graph_"))

    def test_graph_mode_threaded_through_write_chunks(self):
        import tempfile
        import os
        from chunker.chunker import Chunker, write_chunks_to_file
        from chunker.document import Document

        doc = Document.create(
            source="https://example.com",
            title="Test",
            content="First.\n\nSecond.\n\nThird.",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "chunks.json")
            count = write_chunks_to_file([doc], out, graph_mode=True)
            self.assertGreaterEqual(count, 1)
            with open(out, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertGreaterEqual(len(lines), 1)
            import json
            first = json.loads(lines[0])
            self.assertIn("unit_ids", first)
            self.assertIn("graph_id", first)


class ChunkGraphKeywordEdgesTest(unittest.TestCase):
    def test_keyword_edges_link_shared_topics(self):
        from chunker.graph import ChunkGraph, TextUnit

        graph = ChunkGraph()
        units = [
            TextUnit(unit_id="u0", document_id="d1", text="neural networks learn via gradient descent optimization", index=0),
            TextUnit(unit_id="u1", document_id="d1", text="gradient descent optimizes the neural network loss", index=1),
            TextUnit(unit_id="u2", document_id="d1", text="the weather is sunny and warm today", index=2),
        ]
        for unit in units:
            graph.add_unit(unit)
        graph.build_keyword_edges(units, min_shared=2)

        types = [e.edge_type for e in graph.edges]
        self.assertIn("keyword", types)
        pairs = {(e.source, e.target) for e in graph.edges}
        self.assertTrue(pairs & {("u0", "u1"), ("u1", "u0")})
        # unrelated weather unit should not be linked to the ML units
        self.assertFalse(pairs & {("u0", "u2"), ("u2", "u0"), ("u1", "u2"), ("u2", "u1")})


class ChunkGraphAdjacencyTest(unittest.TestCase):
    def test_create_graph_chunks_stores_connected_chunk_ids(self):
        from chunker.chunker import Chunker
        from chunker.document import Document

        paras = []
        for i in range(4):
            paras.append(f"Neural networks learn patterns using gradient descent. The loss function measures error. Training improves the learning rate. Topic block {i} deep learning.")
        for i in range(4):
            paras.append(f"Climate change raises global temperatures and sea levels. Polar ice melts as warming accelerates. Coastal cities face flooding. Topic block {i} environment.")
        doc = Document.create(source="https://example.com", title="T", content="\n\n".join(paras))
        chunks, graph = Chunker().create_graph_chunks(doc)

        for chunk in chunks:
            self.assertIn("connected_chunk_ids", chunk.metadata)
            self.assertIsInstance(chunk.metadata["connected_chunk_ids"], list)

        linked = [c for c in chunks if c.metadata["connected_chunk_ids"]]
        self.assertTrue(linked, "expected at least one chunk with graph connections")
        for chunk in linked:
            for entry in chunk.metadata["connected_chunk_ids"]:
                self.assertIn("chunk_id", entry)
                self.assertIn("weight", entry)


class ChunkerConfigIntegrationTest(unittest.TestCase):
    def test_graph_config_defaults(self):
        from utils.config import (
            get_graph_enabled,
            get_graph_semantic_threshold,
            get_graph_community_resolution,
            get_graph_expansion_hops,
            get_graph_decay,
        )

        self.assertFalse(get_graph_enabled())
        self.assertAlmostEqual(get_graph_semantic_threshold(), 0.75)
        self.assertAlmostEqual(get_graph_community_resolution(), 1.0)
        self.assertEqual(get_graph_expansion_hops(), 2)
        self.assertAlmostEqual(get_graph_decay(), 0.5)


if __name__ == "__main__":
    unittest.main()
