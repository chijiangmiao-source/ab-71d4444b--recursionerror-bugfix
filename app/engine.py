"""Shared packed parse forest (SPPF) engine.

Given a validated :class:`~app.grammar.Grammar` and its input token
stream the engine returns exactly one verdict:

* ``REJECTED``            -- no finite derivation of the input exists,
* ``UNIQUE_ACCEPTED``     -- exactly one finite derivation tree,
* ``AMBIGUOUS_ACCEPTED``  -- at least two distinct finite derivation
                             trees; two are returned, chosen by a stable
                             rule over production-id sequences.

Ambiguity is decided without enumerating derivations.  An Earley chart
constructs a binary SPPF (Scott/Johnstone style):

* symbol nodes    ``('s', X, i, j)``           -- a terminal leaf or a
                                                   packed nonterminal;
* intermediate nodes ``('i', pid, k, i, j)``   -- the prefix
                                                   ``rhs[:k]`` of one
                                                   production spanning
                                                   ``[i, j]``;
* a binary family ``(left_intermediate, right_symbol_node)`` packs one
  way of extending a production prefix by its next symbol.

Every node stores its packed families; the number of distinct trees
under any node is computed as a saturated value in ``{0, 1, 2}`` (2 =
"two or more"), so ambiguity is detected with polynomial work and
space regardless of how many derivations exist.

A static pass runs first: the start symbol must be generating (derive
some finite terminal string), and any reachable *non-consuming cycle*
(a loop of epsilon productions that could expand forever without
reading a token) is rejected outright, so infinite expansion can never
be presented as evidence.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from .grammar import Grammar

# Verdict codes.
REJECTED = "REJECTED"
UNIQUE = "UNIQUE_ACCEPTED"
AMBIGUOUS = "AMBIGUOUS_ACCEPTED"

# Rejection reason codes.
START_UNPRODUCTIVE = "START_UNPRODUCTIVE"
INPUT_NOT_ACCEPTED = "INPUT_NOT_ACCEPTED"
NONCONSUMING_CYCLE = "NONCONSUMING_CYCLE"

# Node key shapes:
#   ('s', name, i, j)               symbol node (terminal leaf or NT)
#   ('i', pid, k, i, j)             intermediate node
SKey = Tuple[Any, ...]
IKey = Tuple[Any, ...]
Family = Tuple[Optional[IKey], SKey]  # (left intermediate, right symbol)


class EngineError(Exception):
    """Static rejection discovered before/without forest construction."""

    def __init__(self, reason: str, detail: str, extra: Optional[dict] = None) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.extra = extra or {}


# ---------------------------------------------------------------------------
# Static grammar analysis
# ---------------------------------------------------------------------------


def _fixed_point(initial: set, step) -> set:
    cur = set(initial)
    while True:
        nxt = step(cur)
        if nxt == cur:
            return cur
        cur = nxt


def static_analysis(g: Grammar) -> None:
    """Reject unproductive start symbols and reachable null cycles."""
    nts = set(g.nonterminals)

    # --- generating: A derives some finite terminal string -------------
    def gen_step(known: set) -> set:
        out = set(known)
        for p in g.productions:
            if p.lhs in out:
                continue
            if all((s in nts and s in out) or s not in nts for s in p.rhs):
                out.add(p.lhs)
        return out

    generating = _fixed_point(set(), gen_step)
    if g.start not in generating:
        raise EngineError(
            START_UNPRODUCTIVE,
            f"起始符号 {g.start!r} 无法派生出任何有限终结符串"
            "（不存在全部由可生成符号构成的产生式链；请检查无终结出口的非终结符）",
        )

    # --- finite epsilon ("useful"): A has a FINITE derivation of ε -----
    # R = { A | some A -> beta with every symbol of beta in R }
    def useful_step(useful: set) -> set:
        out = set(useful)
        for p in g.productions:
            if p.lhs in out:
                continue
            if all(s in nts and s in useful for s in p.rhs):
                out.add(p.lhs)
        return out

    useful = _fixed_point(set(), useful_step)

    # Non-consuming edges A -> B: a production A -> beta with NO terminal
    # on its RHS, and every symbol other than B has a *finite* epsilon
    # derivation.  B itself is not required to be nullable: A -> A alone
    # gives A => A => A ... forever without consuming a token, and that
    # is exactly the infinite expansion we must refuse.  A directed cycle
    # among these edges is a reachable non-consuming cycle.
    edges: Dict[str, List[Tuple[str, int]]] = {}
    for p in g.productions:
        if any(s not in nts for s in p.rhs):
            continue  # a terminal obliges consumption; no null edges
        # Exclusion is positional: in S -> S S each occurrence in turn is
        # the edge target and the *other* occurrence must be ε-useful.
        for idx, b in enumerate(p.rhs):
            others = [s for k, s in enumerate(p.rhs) if k != idx]
            if all(s in useful for s in others):
                edges.setdefault(p.lhs, []).append((b, p.id))
        edges.setdefault(p.lhs, [])

    # Ordinary reachability from the start symbol (any grammar edge).
    reachable = {g.start}
    q = deque([g.start])
    while q:
        a = q.popleft()
        for p in g.productions_of(a):
            for s in p.rhs:
                if s in nts and s not in reachable:
                    reachable.add(s)
                    q.append(s)

    color: Dict[str, int] = {}  # 0 white / 1 grey / 2 black
    stack: List[str] = []
    chosen_pid: Dict[Tuple[str, str], int] = {}

    def dfs(a: str) -> Optional[List[str]]:
        color[a] = 1
        stack.append(a)
        for b, pid in sorted(edges.get(a, []), key=lambda e: (e[0], e[1])):
            if b not in reachable:
                continue
            chosen_pid.setdefault((a, b), pid)
            if color.get(b, 0) == 0:
                found = dfs(b)
                if found is not None:
                    return found
            elif color.get(b) == 1:
                idx = stack.index(b)
                return stack[idx:] + [b]
        stack.pop()
        color[a] = 2
        return None

    for root in sorted(reachable):
        if color.get(root, 0) == 0:
            cyc = dfs(root)
            if cyc is not None:
                pids = [chosen_pid[(u, v)] for u, v in zip(cyc, cyc[1:])]
                detail = (
                    "检测到可达的不消费词元循环（可沿 ε 产生式无限展开而不消费任何词元），"
                    f"已拒绝：{' -> '.join(cyc)}；各步产生式编号依次为 {pids}。"
                    "请打断该零消费环（例如令环上某条产生式至少消费一个词元）"
                )
                raise EngineError(
                    NONCONSUMING_CYCLE,
                    detail,
                    {"cycle": cyc, "production_ids": pids},
                )


# ---------------------------------------------------------------------------
# SPPF construction
# ---------------------------------------------------------------------------


def build_forest(g: Grammar) -> Tuple[Dict[SKey, List[IKey]], Dict[IKey, set]]:
    """Build the binary SPPF; returns ``(symbol_families, inter_families)``.

    * ``symbol_families[S]`` -- list of intermediate nodes completing
      nonterminal symbol node S (terminals have no entry);
    * ``inter_families[I]``  -- set of ``(left, right)`` binary families.

    Earley states ``(pid, start, dot, pos)`` are stored once each; all
    packing happens in the forest, so state count is ``O(|G| n^2)`` and
    the whole structure stays polynomial (at most ``O(|G| n^3)``
    families).
    """
    tokens = g.tokens
    n = len(tokens)

    symbol_families: Dict[SKey, List[IKey]] = {}
    inter_families: Dict[IKey, set] = {}

    states: set = set()
    # waiters[(B, pos)] = states parked with the dot on B at pos
    waiters: Dict[Tuple[str, int], set] = {}
    # Every symbol node ever completed, keyed by the event it can fire.
    completed_snodes: Dict[Tuple[str, int], List[Tuple[SKey, int]]] = {}

    work: deque = deque()

    def inter_add(key: IKey, family: Family) -> None:
        bucket = inter_families.get(key)
        if bucket is None:
            bucket = set()
            inter_families[key] = bucket
        if family not in bucket:
            bucket.add(family)

    def complete(skey: SKey, ikey: IKey) -> None:
        fams = symbol_families.get(skey)
        if fams is None:
            fams = []
            symbol_families[skey] = fams
            work.append(("snode", skey))
            _, name, i, j = skey
            completed_snodes.setdefault((name, i), []).append((skey, j))
        if ikey not in fams:
            fams.append(ikey)

    def add_state(key: Tuple[int, int, int, int]) -> None:
        if key not in states:
            states.add(key)
            work.append(("state", key))

    def advance(waiter: Tuple[int, int, int, int], child: SKey, j: int) -> None:
        pid, start, dot, _x = waiter
        new_i = ("i", pid, dot + 1, start, j)
        left = None if dot == 0 else ("i", pid, dot, start, _x)
        inter_add(new_i, (left, child))
        add_state((pid, start, dot + 1, j))

    # Seed: predict the start symbol at position 0.
    for p in g.productions_of(g.start):
        add_state((p.id, 0, 0, 0))

    while work:
        kind, key = work.popleft()

        if kind == "snode":
            _, name, i, j = key
            for waiter in list(waiters.get((name, i), ())):
                advance(waiter, key, j)
            continue
        pid, start, dot, pos = key
        prod = g.by_id[pid]

        if dot == len(prod.rhs):
            # Completed production: finish the symbol node.
            complete(("n", prod.lhs, start, pos),
                     ("i", pid, dot, start, pos))
            continue

        sym = prod.rhs[dot]
        if g.is_nt.get(sym):
            parked = waiters.setdefault((sym, pos), set())
            already_parked = key in parked
            parked.add(key)
            if not already_parked:
                # Predict.  Duplicate seeds (left/epsilon recursion) are
                # ignored by add_state, so prediction always terminates.
                for q in g.productions_of(sym):
                    add_state((q.id, pos, 0, pos))
            # Nodes completed before this waiter parked must fire now;
            # otherwise their "snode" event was already consumed.
            for skey, j in completed_snodes.get((sym, pos), ()):
                advance(key, skey, j)
        else:
            # Scan exactly one input token.
            if pos < n and tokens[pos] == sym:
                leaf = ("t", sym, pos, pos + 1)
                new_i = ("i", pid, dot + 1, start, pos + 1)
                left = None if dot == 0 else ("i", pid, dot, start, pos)
                inter_add(new_i, (left, leaf))
                add_state((pid, start, dot + 1, pos + 1))

    return symbol_families, inter_families


# ---------------------------------------------------------------------------
# Verdict and stable tree extraction
# ---------------------------------------------------------------------------


def _is_terminal(skey: SKey) -> bool:
    return skey[0] == "t"


def analyze_forest(g: Grammar, symbol_families: Dict[SKey, List[IKey]],
                   inter_families: Dict[IKey, set]) -> dict:
    tokens = g.tokens
    n = len(tokens)
    root = ("n", g.start, 0, n)
    if root not in symbol_families:
        hint = ""
        alphabet = {s for p in g.productions for s in p.rhs if not g.is_nt.get(s)}
        for idx, tok in enumerate(tokens):
            if tok not in alphabet:
                hint = f"；首个不被任何产生式右部引用的词元为位置 {idx} 的 {tok!r}"
                break
        raise EngineError(
            INPUT_NOT_ACCEPTED,
            f"输入词元序列不被起始符号 {g.start!r} 接受：共享森林中不存在"
            f" {g.start} 覆盖 [0,{n}] 的完整派生节点{hint}",
        )

    # Recipe shapes (immutable, structurally shared):
    #   ("T", terminal_skey)
    #   ("N", skey, pid, (child_recipe, ...))
    Recipe = tuple
    best_s: Dict[SKey, List[Tuple[Tuple[int, ...], Recipe]]] = {}
    best_i: Dict[IKey, List[Tuple[Tuple[int, ...], Tuple[Recipe, ...]]]] = {}

    # Every node's value depends only on its forest children, so the DAG
    # is evaluated with an explicit work stack (postorder DFS) rather than
    # Python recursion.  This matters for the protocol's *wide*
    # productions: binarization turns ``A -> X1...Xm`` into an m-deep
    # left-leaning chain of intermediate nodes, and a legal RHS may be far
    # longer than the interpreter's recursion limit (e.g. 1200 nullable
    # symbols against the empty input).  Deep parse trees get the same
    # treatment for free.  Memoization still computes each node once.
    #
    # Work items are ("n", kind, key) to visit a node and
    # ("d", kind, key) emitted beforehand to combine its children once
    # every descendant visit has been resolved.
    #   progress[(kind, key)]: 1 = visited, awaiting the combine marker
    def merge(cands):
        """Deduplicate by sequence, sort ascending, keep the two smallest."""
        by_seq = {}
        for seq, payload in cands:
            if seq not in by_seq:
                by_seq[seq] = payload
        return [(seq, by_seq[seq]) for seq in sorted(by_seq)[:2]]

    def base_s(skey: SKey) -> bool:
        if _is_terminal(skey):
            best_s[skey] = [((), ("T", skey))]
            return True
        return False

    def base_i(ikey: IKey) -> bool:
        if ikey[2] == 0:  # empty production prefix: one way, no children
            best_i[ikey] = [((), ())]
            return True
        return False

    progress: Dict[Tuple[str, tuple], int] = {}

    def children_of(kind: str, key):
        if kind == "s":
            for ik in symbol_families[key]:
                yield "i", ik
        else:
            for left, right in inter_families[key]:
                yield "s", right
                if left is not None:
                    yield "i", left

    work: list = [("n", "s", root)]
    while work:
        tag, kind, key = work.pop()

        if tag == "n":
            done = best_s if kind == "s" else best_i
            if key in done:
                continue
            if kind == "s" and base_s(key):
                continue
            if kind == "i" and base_i(key):
                continue
            marker = (kind, key)
            if progress.get(marker) == 1:
                # Reaching an in-progress node from one of its descendants
                # is a circular forest dependency; the static pass rejects
                # every such grammar, so this is only a safety net.
                raise EngineError(
                    NONCONSUMING_CYCLE,
                    f"内部错误：森林提取阶段在节点 {key} 遇到循环依赖",
                )
            progress[marker] = 1
            work.append(("d", kind, key))
            for ckind, ckey in children_of(kind, key):
                cdone = best_s if ckind == "s" else best_i
                if ckey not in cdone:
                    work.append(("n", ckind, ckey))
            continue

        # tag == "d": all descendants are memoized; combine them.
        progress.pop((kind, key), None)
        if kind == "s":
            skey = key
            cands = []
            for ik in symbol_families[skey]:
                pid = ik[1]
                for seq, kids in best_i[ik]:
                    cands.append(((pid,) + seq, ("N", skey, pid, tuple(kids))))
            best_s[skey] = merge(cands)
        else:
            ikey = key
            cands = []
            for left, right in inter_families[ikey]:
                right_opts = best_s[right]
                left_opts = best_i[left] if left is not None else [((), ())]
                for lseq, lkids in left_opts:
                    for rseq, rrecipe in right_opts:
                        cands.append((lseq + rseq, tuple(lkids) + (rrecipe,)))
            best_i[ikey] = merge(cands)

    options = best_s[root]
    ambiguous = len(options) >= 2

    def render(recipe: Recipe) -> dict:
        # Iterative tree materialization: a legal derivation can be deeper
        # than the interpreter recursion limit, so no Python call may nest
        # per tree node.  Each task fills a holder dict that the parent
        # already references (placeholders pushed in reverse child order).
        root_obj: dict = {}
        tasks: List[Tuple[Recipe, dict]] = [(recipe, root_obj)]
        while tasks:
            cur, holder = tasks.pop()
            if cur[0] == "T":
                _, skey = cur
                holder["token"] = skey[1]
                holder["span"] = [skey[2], skey[3]]
                continue
            _, skey, pid, kids = cur
            children = [{} for _ in kids]
            holder["symbol"] = skey[1]
            holder["production"] = pid
            holder["span"] = [skey[2], skey[3]]
            holder["children"] = children
            for k, child_holder in zip(reversed(kids), reversed(children)):
                tasks.append((k, child_holder))
        return root_obj

    first_seq, first_recipe = options[0]
    result: Dict[str, Any] = {
        "verdict": AMBIGUOUS if ambiguous else UNIQUE,
        "input_length": n,
    }

    if not ambiguous:
        result["tree"] = render(first_recipe)
        result["production_sequence"] = list(first_seq)
        return result

    second_seq, second_recipe = options[1]
    result["trees"] = {
        "first": render(first_recipe),
        "second": render(second_recipe),
    }
    result["production_sequences"] = {
        "first": list(first_seq),
        "second": list(second_seq),
    }
    result["selection_rule"] = (
        "先序产生式编号序列对有限派生树构成稳定全序；"
        "first 取该全序下字典序最小的序列，second 取字典序次小的序列；"
        "同一序列唯一对应一棵具体派生树（共享森林按序列去重，不枚举全部推导）"
    )
    return result


def analyze(g: Grammar) -> dict:
    """Full static + SPPF analysis.  Raises :class:`EngineError`."""
    static_analysis(g)
    symbol_families, inter_families = build_forest(g)
    return analyze_forest(g, symbol_families, inter_families)
