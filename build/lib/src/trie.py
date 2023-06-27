import math
from typing import Optional, list


class TrieNode:

  def __init__(self) -> None:
    self.children: dict[int, TrieNode] = {}
    self.is_end_of_game: bool = False
    self.game_count: int = 0


class Trie:

  def __init__(self) -> None:
    self.root: TrieNode = TrieNode()

  def insert(self, game: list[int]) -> None:
    node = self.root
    for move in game:
      if move not in node.children:
        node.children[move] = TrieNode()
      node.game_count += 1
      node = node.children[move]
    node.is_end_of_game = True

  def search(self, game: list[int]) -> None:
    node = self.root
    for move in game:
      if move not in node.children:
        return False
      node = node.children[move]
    return node.is_end_of_game

  def starts_with(self, initial_moves: list[int]) -> bool:
    node = self.root
    for move in initial_moves:
      if move not in node.children:
        return False
      node = node.children[move]
    return True

  def partial_game_entropy(self, initial_moves: list[int]) -> float:
    node = self.root
    for move in initial_moves:
      if move not in node.children:
        raise ValueError(
          f"Game prefix {initial_moves} does not exist in Trie.")
      node = node.children[move]

    entropy = 0
    for child_node in node.children.values():
      # Calculate probability of moving to the child node
      prob = child_node.game_count / node.game_count
      entropy -= prob * math.log2(prob)

    return entropy

  def entropy(self, node: TrieNode) -> float:
    entropy = 0
    for child_node in node.children.values():
      prob = child_node.game_count / node.game_count
      if prob > 0:
        entropy -= prob * math.log2(prob)

    return entropy

  def total_entropy(self, node: Optional[TrieNode] = None) -> float:
    if node is None:
      node = self.root

    total_entropy = self.entropy(node)

    for child in node.children.values():
      total_entropy += self.total_entropy(child)

    return total_entropy

  def average_entropy(self) -> float:
    total_tokens = self.root.game_count * 10
    return self.total_entropy() / total_tokens
