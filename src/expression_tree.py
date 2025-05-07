# expression_tree.py

import math
import re

class ExpressionTree:
    """
    Represents a mathematical expression as a binary tree.

    Features:
    - Build from string expression (supports +, -, *, /, ^).
    - Calculate structural distance between two expression trees.
    - Compute expression complexity (number of nodes).
    """

    def __init__(self, root=None):
        """
        Initialize the expression tree with an optional root node.
        """
        self.root = root

    def is_empty(self):
        """
        Check if the tree is empty.
        """
        return self.root is None

    def build_from_string(self, expression_string):
        """
        Parse and construct an expression tree from a given expression string.

        Parameters:
        ----------
        expression_string : str
            A mathematical expression in infix notation.
        """
        self.tokens = re.findall(r"(\d+\.?\d*|\+|-|\*|/|\^|\(|\)|\w+)", expression_string)
        self.current_token_index = 0
        self.root = self._parse_expression()

    def _get_next_token(self):
        if self.current_token_index < len(self.tokens):
            token = self.tokens[self.current_token_index]
            self.current_token_index += 1
            return token
        return None

    def _peek_next_token(self):
        if self.current_token_index < len(self.tokens):
            return self.tokens[self.current_token_index]
        return None

    def _parse_expression(self):
        left = self._parse_term()
        while self._peek_next_token() in ('+', '-'):
            op = self._get_next_token()
            right = self._parse_term()
            left = (op, left, right)
        return left

    def _parse_term(self):
        left = self._parse_factor()
        while self._peek_next_token() in ('*', '/', '^'):
            op = self._get_next_token()
            right = self._parse_factor()
            left = (op, left, right)
        return left

    def _parse_factor(self):
        token = self._get_next_token()
        if token is None:
            raise ValueError("Invalid expression: unexpected end")

        if re.match(r"^\d+(\.\d+)?$", token) or re.match(r"^[a-zA-Z_]\w*$", token):
            return float(token) if token.replace('.', '', 1).isdigit() else token

        if token == '(':
            expr = self._parse_expression()
            closing = self._get_next_token()
            if closing != ')':
                raise ValueError("Expected ')'")
            return expr

        raise ValueError(f"Unrecognized token: {token}")

    def distance(self, other_tree):
        """
        Compute a structural distance between this tree and another.

        Parameters:
        ----------
        other_tree : ExpressionTree
            The other tree to compare with.

        Returns:
        -------
        float
            A distance score between 0 (identical) and higher values (more different).
        """
        return self._distance_helper(self.root, other_tree.root)

    def _distance_helper(self, node1, node2):
        if node1 is None and node2 is None:
            return 0.0
        if node1 is None or node2 is None:
            return 1.0
        if isinstance(node1, float) and isinstance(node2, float):
            return abs(node1 - node2) * 0.1
        if isinstance(node1, str) and isinstance(node2, str):
            return 0.0 if node1 == node2 else 0.5
        if isinstance(node1, (float, str)) and isinstance(node2, tuple):
            return 1.0
        if isinstance(node1, tuple) and isinstance(node2, (float, str)):
            return 1.0

        op1, left1, right1 = node1
        op2, left2, right2 = node2
        if op1 != op2:
            return 1.0

        if op1 in ('+', '*'):
            dist1 = self._distance_helper(left1, left2) + self._distance_helper(right1, right2)
            dist2 = self._distance_helper(left1, right2) + self._distance_helper(right1, left2)
            return min(dist1, dist2)

        return self._distance_helper(left1, left2) + self._distance_helper(right1, right2)

    def compute_complexity(self):
        """
        Calculate the number of nodes in the expression tree.

        Returns:
        -------
        int
            The total number of nodes (operators, values, variables).
        """
        return self._compute_complexity_helper(self.root)

    def _compute_complexity_helper(self, node):
        if node is None:
            return 0
        if isinstance(node, (float, str)):
            return 1
        if isinstance(node, tuple):
            _, left, right = node
            return 1 + self._compute_complexity_helper(left) + self._compute_complexity_helper(right)
        return 0

    def __str__(self):
        """
        Get string representation of the expression.
        """
        return self._to_string(self.root)

    def _to_string(self, node):
        if node is None:
            return ""
        if isinstance(node, float):
            return str(node)
        if isinstance(node, str):
            return node
        if isinstance(node, tuple) and len(node) == 3:
            op, left, right = node
            return f"({self._to_string(left)} {op} {self._to_string(right)})"
        return str(node)
