from types import SimpleNamespace

import mp_export


def test_compact_birrt_tree_starts_small():
  tree = SimpleNamespace()
  mp_export._init_compact_bidir_tree(
      tree, SimpleNamespace(robot_dims={"a1": 7, "a2": 7, "a3": 7, "a4": 7}))

  cap = mp_export._BIRRT_INITIAL_TREE_CAPACITY
  assert tree.batch_subtree.shape == (cap, 28)
  assert tree.batch_subtree_b.shape == (cap, 28)
  assert tree.node_ids_subtree.shape == (cap,)
  assert tree.node_ids_subtree_b.shape == (cap,)
  assert tree.subtree == tree.subtree_b == {}
  assert tree.connected is False
