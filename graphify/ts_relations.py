"""System 7 patch: TypeScript/TSX/JSX relation edges added as an extraction post-pass.

Enabled only when ``GRAPHIFY_TS_RELATIONS`` is set to ``1``/``true``/``yes``.
With the flag off, ``add_ts_relations`` is never called and output is byte-for-byte
the unpatched Graphify 0.9.96 output.

Everything is derived from the AST and the import structure of each file. Node ids
are never minted here except for unbound inline GraphQL operations; edges point at
node ids that already exist in the Graphify node list.

Relations added (same edge shape Graphify emits):
  renders, calls_hook, uses_store, route_renders, tested_by, uses_gql_operation
and a ``type_only`` boolean on every JS/TS ``imports`` / ``imports_from`` edge.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

JSTS_SUFFIXES = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
_TEST_RE = re.compile(r"\.(test|spec)\.[cm]?[jt]sx?$", re.I)
_HOOK_RE = re.compile(r"^use[A-Z0-9]")
_ROUTE_FACTORY_RE = re.compile(r"^create(File|Lazy|Root)?Route(WithContext)?$|^createLazyFileRoute$")
_ROUTE_COMPONENT_KEYS = {"component", "pendingComponent", "errorComponent", "notFoundComponent",
                         "Component", "ErrorBoundary", "element"}
_STORE_FACTORIES = {"create", "createStore", "createWithEqualityFn"}
_GQL_TAGS = {"gql", "graphql"}
_GQL_OP_RE = re.compile(r"\b(?:query|mutation|subscription|fragment)\s+([A-Za-z_]\w*)")
_FUNC_TYPES = {"function_declaration", "generator_function_declaration", "class_declaration",
               "abstract_class_declaration"}
_MOCK_OBJS = {"vi", "jest"}
_MOCK_METHODS = {"mock", "doMock", "importActual", "requireActual", "importMock", "unmock"}


def enabled() -> bool:
    return os.environ.get("GRAPHIFY_TS_RELATIONS", "").strip().lower() in ("1", "true", "yes", "on")


def _txt(n) -> str:
    return n.text.decode("utf-8", "replace") if n is not None else ""


def _str_value(n) -> str | None:
    if n is None or n.type not in ("string", "template_string"):
        return None
    if n.type == "template_string" and any(c.type == "template_substitution" for c in n.children):
        return None
    return _txt(n).strip("'\"` ")


class _FileInfo:
    __slots__ = ("rel", "abs", "tree", "src", "imports", "exports", "reexports", "stars",
                 "stmts", "stores", "gql_consts", "is_test")

    def __init__(self, rel, abs_):
        self.rel = rel
        self.abs = abs_
        self.tree = None
        self.src = b""
        # local name -> (module_raw, imported_name | "default" | "*", type_only)
        self.imports: dict = {}
        self.exports: dict = {}      # exported name -> local name
        self.reexports: dict = {}    # exported name -> (module_raw, name)
        self.stars: list = []        # modules re-exported with export *
        self.stmts: list = []        # (line, module_raw, type_only)
        self.stores: set = set()     # local names created by zustand create(...)
        self.gql_consts: set = set()
        self.is_test = bool(_TEST_RE.search(rel))


def add_ts_relations(paths, root, all_nodes, all_edges) -> dict:
    """Append the new relations to ``all_edges`` in place. Returns per-relation counts."""
    from tree_sitter import Language, Parser
    import tree_sitter_typescript as _ts
    import tree_sitter_javascript as _js
    from graphify.extractors.resolution import _resolve_js_import_target
    from graphify.ids import make_id as _mk

    parsers = {
        "tsx": Parser(Language(_ts.language_tsx())),
        "ts": Parser(Language(_ts.language_typescript())),
        "js": Parser(Language(_js.language())),
    }

    # ---- indexes over the already-final node list -------------------------------------
    file_node: dict[str, str] = {}
    syms: dict[str, dict[str, str]] = {}
    for n in all_nodes:
        sf = n.get("source_file")
        if not sf or n.get("file_type") != "code":
            continue
        label = n.get("label") or ""
        if label == os.path.basename(sf) and n.get("source_location") == "L1":
            file_node.setdefault(sf, n["id"])
            continue
        name = label[:-2] if label.endswith("()") else label
        if not name or name.startswith(".") or "." in name and not name.replace("_", "").isalnum():
            continue
        syms.setdefault(sf, {}).setdefault(name, n["id"])
    node_ids = {n["id"] for n in all_nodes}

    infos: dict[str, _FileInfo] = {}
    root = Path(root)
    for p in paths:
        p = Path(p)
        if p.suffix.lower() not in JSTS_SUFFIXES:
            continue
        try:
            rel = p.resolve().relative_to(root.resolve()).as_posix()
        except (ValueError, OSError):
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                continue
        if rel not in file_node:
            continue
        infos[rel] = _FileInfo(rel, p)

    # ---- pass 1: parse + collect imports / exports / stores / gql constants -------------
    for fi in infos.values():
        try:
            fi.src = fi.abs.read_bytes()
        except OSError:
            continue
        suf = fi.abs.suffix.lower()
        key = "tsx" if suf == ".tsx" else ("ts" if suf in (".ts", ".mts", ".cts") else "js")
        try:
            fi.tree = parsers[key].parse(fi.src)
        except Exception:
            fi.tree = None
            continue
        _collect_module_info(fi)

    resolve_cache: dict = {}

    def resolve_module(fi, raw):
        k = (fi.rel, raw)
        if k in resolve_cache:
            return resolve_cache[k]
        out = None
        try:
            r = _resolve_js_import_target(raw, str(fi.abs))
        except Exception:
            r = None
        if r is not None:
            rp = r[1]
            try:
                if rp is not None and rp.is_file():
                    try:
                        rrel = rp.resolve().relative_to(root.resolve()).as_posix()
                    except ValueError:
                        rrel = rp.relative_to(root).as_posix()
                    if rrel in file_node:
                        out = rrel
            except (OSError, ValueError):
                out = None
        resolve_cache[k] = out
        return out

    def export_target(frel, name, depth=0):
        """Resolve an exported name of file ``frel`` to (file_rel, local_name, node_id)."""
        if depth > 6 or frel not in infos:
            return None
        fi = infos[frel]
        local = fi.exports.get(name)
        if local is not None:
            nid = syms.get(frel, {}).get(local)
            if nid:
                return (frel, local, nid)
            # `import X from './x'; export { X }` style pass-through
            imp = fi.imports.get(local)
            if imp:
                return import_target(fi, local, depth + 1)
        if name in fi.reexports:
            mod, nm = fi.reexports[name]
            t = resolve_module(fi, mod)
            if t:
                return export_target(t, nm, depth + 1)
        if name != "default":
            nid = syms.get(frel, {}).get(name)
            if nid and name in fi.exports:
                return (frel, name, nid)
            for mod in fi.stars:
                t = resolve_module(fi, mod)
                if t:
                    r = export_target(t, name, depth + 1)
                    if r:
                        return r
        return None

    def import_target(fi, local, depth=0):
        imp = fi.imports.get(local)
        if not imp:
            return None
        mod, nm, _t = imp
        t = resolve_module(fi, mod)
        if not t:
            return None
        if nm == "*":
            return (t, "*", file_node[t])
        r = export_target(t, nm, depth)
        if r:
            return r
        return (t, nm, file_node[t])  # barrel / unresolvable symbol: fall back to the file

    def resolve_ident(fi, name, member=None):
        """Identifier -> (file_rel, local_name, node_id, via_import) or None.
        ``member`` resolves ``NS.member`` through a namespace import."""
        if name in fi.imports:
            r = import_target(fi, name)
            if r and r[1] == "*" and member:
                r2 = export_target(r[0], member)
                if r2:
                    return r2 + (True,)
            return (r + (True,)) if r else None
        nid = syms.get(fi.rel, {}).get(name)
        if nid:
            return (fi.rel, name, nid, False)
        return None

    def is_store(frel, local):
        t = infos.get(frel)
        return bool(t and local in t.stores)

    def is_gql(frel, local):
        t = infos.get(frel)
        return bool(t and local in t.gql_consts)

    # ---- pass 2: emit edges ----------------------------------------------------------
    new_edges: list = []
    seen: set = set()
    counts: dict = {}
    new_nodes: list = []

    def emit(src, tgt, rel, ctx, fi, line, conf="EXTRACTED", score=1.0):
        if not src or not tgt or src == tgt:
            return
        k = (src, tgt, rel)
        if k in seen:
            return
        seen.add(k)
        new_edges.append({
            "source": src, "target": tgt, "relation": rel, "_origin": "ast",
            "confidence": conf, "confidence_score": score, "context": ctx,
            "source_file": fi.rel, "source_location": f"L{line}", "weight": 1.0,
        })
        counts[rel] = counts.get(rel, 0) + 1

    for fi in infos.values():
        if fi.tree is None:
            continue
        fnode = file_node[fi.rel]
        fsyms = syms.get(fi.rel, {})

        # tested_by: test file -> source file it imports / mocks
        if fi.is_test:
            for line, raw, tonly in fi.stmts:
                if tonly:
                    continue
                t = resolve_module(fi, raw)
                if t and t != fi.rel and not infos[t].is_test:
                    emit(fnode, file_node[t], "tested_by", "test_import", fi, line)
            for line, raw in _mock_specifiers(fi.tree.root_node):
                t = resolve_module(fi, raw)
                if t and t != fi.rel and not infos[t].is_test:
                    emit(fnode, file_node[t], "tested_by", "test_mock", fi, line)

        # AST walk with enclosing top-level symbol tracking
        stack = [(fi.tree.root_node, fnode, True)]
        while stack:
            node, enc, top = stack.pop()
            t = node.type
            enc_here = enc
            child_top = False
            if t == "program":
                child_top = True
            elif top and t == "export_statement":
                child_top = True
            elif top and t in _FUNC_TYPES:
                nm = node.child_by_field_name("name")
                nid = fsyms.get(_txt(nm)) if nm is not None else None
                if nid:
                    enc_here = nid
            elif top and t in ("lexical_declaration", "variable_declaration"):
                child_top = True
            elif top and t == "variable_declarator":
                nm = node.child_by_field_name("name")
                if nm is not None and nm.type == "identifier":
                    nid = fsyms.get(_txt(nm))
                    if nid:
                        enc_here = nid
            line = node.start_point[0] + 1

            if t in ("jsx_opening_element", "jsx_self_closing_element"):
                _on_jsx(node, fi, enc_here, line, resolve_ident, emit, is_gql)
            elif t == "call_expression":
                _on_call(node, fi, enc_here, line, fnode, resolve_ident, is_store, is_gql, emit)
            elif t == "member_expression":
                _on_member(node, fi, enc_here, line, resolve_ident, is_store, is_gql, emit, import_target)
            elif t == "identifier" and node.parent is not None:
                _on_ident(node, fi, enc_here, line, resolve_ident, is_gql, emit)
            elif t == "jsx_attribute":
                _on_jsx_attr(node, fi, enc_here, line, resolve_ident, emit)
            elif t == "object":
                _on_route_object(node, fi, fnode, line, resolve_ident, emit)

            for c in node.children:
                stack.append((c, enc_here, child_top))

        # inline (unbound) gql operations -> operation node
        for (line, opname, enc_line_owner) in _inline_gql_ops(fi.tree.root_node):
            nid = _mk(fi.rel.rsplit(".", 1)[0], "gql", opname)
            if nid in node_ids or any(nn["id"] == nid for nn in new_nodes):
                pass
            else:
                new_nodes.append({"id": nid, "label": opname, "file_type": "code", "source_file": fi.rel,
                                  "source_location": f"L{line}", "_origin": "ast"})
            emit(fnode, nid, "uses_gql_operation", "gql_inline", fi, line)

    # contexts of gql constants: file -> const defined in that file
    for fi in infos.values():
        fnode = file_node[fi.rel]
        for name in fi.gql_consts:
            nid = syms.get(fi.rel, {}).get(name)
            if nid:
                emit(fnode, nid, "uses_gql_operation", "gql_definition", fi, _decl_line(fi, name))

    # ---- type_only on every JS/TS import link -----------------------------------------
    type_lines: dict = {}
    for fi in infos.values():
        for line, raw, tonly in fi.stmts:
            k = (fi.rel, f"L{line}")
            type_lines[k] = type_lines.get(k, True) and tonly
    n_type = 0
    for e in all_edges:
        if e.get("relation") in ("imports", "imports_from") and e.get("source_file") in infos:
            v = type_lines.get((e["source_file"], e.get("source_location")))
            val = bool(e.get("type_only")) or bool(v)
            e["type_only"] = val
            n_type += val
    counts["type_only(true)"] = n_type

    # The new relations are added to the graph here, so the build's dedup and clustering handle
    # them like every other edge. Scope rule A (renders, calls_hook, uses_store, route_renders from
    # test files) is applied to the new edges. Rule B only drops package.json-sourced imports,
    # which this pass does not add.
    kept = []
    for e in new_edges:
        if e["relation"] in _RULE_A_RELS and _is_test_path(e.get("source_file", "")):
            counts["rule_A_dropped"] = counts.get("rule_A_dropped", 0) + 1
            continue
        kept.append(e)
    existing_ids = {n["id"] for n in all_nodes}
    all_nodes.extend(n for n in new_nodes if n["id"] not in existing_ids)
    added, annotated = merge_new_edges(all_edges, kept)
    all_edges.extend(added)
    counts["annotated_existing_pairs"] = annotated
    # GRAPHIFY_TS_OVERLAY_OUT=<file> also writes the new relations to a sidecar, for debugging.
    out = os.environ.get("GRAPHIFY_TS_OVERLAY_OUT")
    if out:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"nodes": new_nodes, "edges": kept, "counts": counts}, f)
    return counts


def merge_new_edges(all_edges: list, new_edges: list) -> tuple[list, int]:
    """Split ``new_edges`` into the edges to add, and record the rest on edges that already exist.

    The graph keeps one edge per pair of nodes, whatever its relation, and a later edge with a different relation
    overwrites an earlier one. So a new relation on a pair that is already linked (by a call, an import or a
    containment) is not added as an edge. It is recorded on the existing edge as ``ts_relations``, and that edge keeps
    its own relation. Direction is ignored, so both orders count as the same pair.

    Returns (edges to add, number of relations recorded on existing edges).
    """
    host_by_pair: dict = {}
    for e in all_edges:
        host_by_pair.setdefault((e.get("source"), e.get("target")), e)
        host_by_pair.setdefault((e.get("target"), e.get("source")), e)
    added: list = []
    annotated = 0
    for e in new_edges:
        pair = (e["source"], e["target"])
        host = host_by_pair.get(pair)
        if host is None:
            added.append(e)
            host_by_pair[pair] = e
            host_by_pair.setdefault((e["target"], e["source"]), e)
            continue
        rels = host.setdefault("ts_relations", [])
        if e["relation"] not in rels:
            rels.append(e["relation"])
        annotated += 1
    return added, annotated


_RULE_A_RELS = {"renders", "calls_hook", "uses_store", "route_renders"}
_RULE_B_RELS = {"imports", "imports_from", "re_exports", "dynamic_import"}
_TEST_DIRS = {"__tests__", "test", "tests", "e2e"}


def _is_test_path(p) -> bool:
    if not p:
        return False
    parts = p.replace("\\", "/").split("/")
    return bool(re.search(r"\.(test|spec)\.", parts[-1])) or any(x in _TEST_DIRS for x in parts[:-1])


def _apply_scope_rules(g: dict) -> dict:
    """Scope rule A (test-file source) and B (package.json source), applied to the merged links
    (Graphify base links and overlay links alike). Counts per rule/relation/origin."""
    nodes = {n["id"]: n for n in g["nodes"]}
    rep = {"rule_A_test_source": {}, "rule_B_package_json_source": {}}
    kept = []
    for l in g["links"]:
        n = nodes.get(l["source"], {})
        sf = n.get("source_file") or l.get("source_file") or ""
        org = "overlay" if l.get("_origin") == "ast" and l["relation"] in _RULE_A_RELS | {"tested_by", "uses_gql_operation"} else "graphify_base"
        rule = None
        if l["relation"] in _RULE_A_RELS and _is_test_path(sf):
            rule = "rule_A_test_source"
        elif l["relation"] in _RULE_B_RELS and (sf.replace("\\", "/").split("/")[-1] == "package.json" or n.get("label") == "package.json"):
            rule = "rule_B_package_json_source"
        if rule:
            d = rep[rule].setdefault(l["relation"], 0)
            rep[rule][l["relation"]] = d + 1
            continue
        kept.append(l)
    g["links"] = kept
    return rep


def merge_overlay(graph_json: str, overlay_json: str) -> dict:
    """Append overlay links (and any new nodes) to a finished graph.json, in place."""
    import json
    g = json.load(open(graph_json))
    ov = json.load(open(overlay_json))
    nodes = {n["id"]: n for n in g["nodes"]}
    for n in ov["nodes"]:
        if n["id"] not in nodes:
            fn = next((m for m in g["nodes"] if m.get("source_file") == n["source_file"]
                       and m.get("source_location") == "L1"), None)
            if fn is not None:
                for k in ("community", "community_name"):
                    if k in fn:
                        n[k] = fn[k]
            n["norm_label"] = n["label"].lower()
            g["nodes"].append(n)
            nodes[n["id"]] = n
    have = {(l["source"], l["target"], l["relation"]) for l in g["links"]}
    added: dict = {}
    for e in ov["edges"]:
        if e["source"] in nodes and e["target"] in nodes and (e["source"], e["target"], e["relation"]) not in have:
            g["links"].append(e)
            added[e["relation"]] = added.get(e["relation"], 0) + 1
    scope = _apply_scope_rules(g)
    json.dump(g, open(graph_json, "w"), separators=(",", ":"))
    json.dump(scope, open(graph_json + ".scope_rules_report.json", "w"), indent=1)
    return added


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "merge":
        print(merge_overlay(sys.argv[2], sys.argv[3]))


def _decl_line(fi, name):
    for n in _iter(fi.tree.root_node):
        if n.type == "variable_declarator":
            nm = n.child_by_field_name("name")
            if nm is not None and _txt(nm) == name:
                return n.start_point[0] + 1
    return 1


def _iter(root):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _collect_module_info(fi):
    root = fi.tree.root_node
    for st in root.children:
        t = st.type
        if t == "import_statement":
            _collect_import(fi, st)
        elif t == "export_statement":
            _collect_export(fi, st)
        # top-level declarations: stores / gql consts (also inside `export const`)
        decl = st
        if t == "export_statement":
            decl = next((c for c in st.children if c.type in ("lexical_declaration", "variable_declaration")), None)
            if decl is None:
                continue
        if decl.type in ("lexical_declaration", "variable_declaration"):
            for vd in decl.children:
                if vd.type != "variable_declarator":
                    continue
                nm = vd.child_by_field_name("name")
                val = vd.child_by_field_name("value")
                if nm is None or nm.type != "identifier" or val is None:
                    continue
                name = _txt(nm)
                if _is_store_init(fi, val):
                    fi.stores.add(name)
                if _is_gql_init(val):
                    fi.gql_consts.add(name)
    # dynamic / side-effect imports for tested_by + type_only bookkeeping are handled elsewhere
    # dynamic import("x") as a runtime dependency statement (not type-only)
    for n in _iter(root):
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None and fn.type == "import":
                args = n.child_by_field_name("arguments")
                s = next((c for c in (args.children if args else []) if c.type == "string"), None)
                v = _str_value(s)
                if v:
                    fi.stmts.append((n.start_point[0] + 1, v, False))


def _collect_import(fi, st):
    stmt_type = any(c.type == "type" and not c.is_named for c in st.children)
    src = next((c for c in st.children if c.type == "string"), None)
    raw = _str_value(src)
    if raw is None:
        return
    clause = next((c for c in st.children if c.type == "import_clause"), None)
    all_type = stmt_type
    if clause is not None and not stmt_type:
        specs = []
        for c in clause.children:
            if c.type == "identifier":
                specs.append(False)
            elif c.type == "namespace_import":
                specs.append(False)
            elif c.type == "named_imports":
                for s in c.children:
                    if s.type == "import_specifier":
                        specs.append(any(x.type == "type" and not x.is_named for x in s.children))
        all_type = bool(specs) and all(specs)
    if clause is not None:
        for c in clause.children:
            if c.type == "identifier":
                fi.imports[_txt(c)] = (raw, "default", stmt_type)
            elif c.type == "namespace_import":
                idn = next((x for x in c.children if x.type == "identifier"), None)
                if idn is not None:
                    fi.imports[_txt(idn)] = (raw, "*", stmt_type)
            elif c.type == "named_imports":
                for s in c.children:
                    if s.type != "import_specifier":
                        continue
                    ids = [x for x in s.children if x.type == "identifier"]
                    if not ids:
                        continue
                    imported = _txt(ids[0])
                    local = _txt(ids[-1])
                    tonly = stmt_type or any(x.type == "type" and not x.is_named for x in s.children)
                    fi.imports[local] = (raw, imported, tonly)
    fi.stmts.append((st.start_point[0] + 1, raw, all_type))


def _collect_export(fi, st):
    src = next((c for c in st.children if c.type == "string"), None)
    raw = _str_value(src)
    is_default = any(c.type == "default" for c in st.children)
    clause = next((c for c in st.children if c.type == "export_clause"), None)
    if raw is not None:
        if clause is None:
            if any(c.type == "*" for c in st.children) or any(c.type == "namespace_export" for c in st.children):
                if not any(c.type == "namespace_export" for c in st.children):
                    fi.stars.append(raw)
        else:
            for s in clause.children:
                if s.type == "export_specifier":
                    ids = [x for x in s.children if x.type == "identifier"]
                    if ids:
                        fi.reexports[_txt(ids[-1])] = (raw, _txt(ids[0]))
        return
    if clause is not None:
        for s in clause.children:
            if s.type == "export_specifier":
                ids = [x for x in s.children if x.type == "identifier"]
                if ids:
                    fi.exports[_txt(ids[-1])] = _txt(ids[0])
        return
    for c in st.children:
        if c.type in _FUNC_TYPES or c.type in ("interface_declaration", "type_alias_declaration",
                                               "enum_declaration"):
            nm = c.child_by_field_name("name")
            if nm is not None:
                if is_default:
                    fi.exports["default"] = _txt(nm)
                fi.exports[_txt(nm)] = _txt(nm)
        elif c.type in ("lexical_declaration", "variable_declaration"):
            for vd in c.children:
                if vd.type == "variable_declarator":
                    nm = vd.child_by_field_name("name")
                    if nm is not None and nm.type == "identifier":
                        fi.exports[_txt(nm)] = _txt(nm)
        elif is_default and c.type == "identifier":
            fi.exports["default"] = _txt(c)
        elif is_default and c.type == "call_expression":
            # export default memo(Foo) / forwardRef(...) -> first identifier argument
            args = c.child_by_field_name("arguments")
            first = next((a for a in (args.children if args else []) if a.type == "identifier"), None)
            if first is not None:
                fi.exports["default"] = _txt(first)


def _root_callee(call):
    fn = call
    while fn is not None and fn.type == "call_expression":
        fn = fn.child_by_field_name("function")
    return fn


def _is_store_init(fi, val):
    if val.type != "call_expression":
        return False
    callee = _root_callee(val)
    if callee is None or callee.type != "identifier":
        return False
    imp = fi.imports.get(_txt(callee))
    return bool(imp and imp[0].startswith("zustand") and imp[1] in _STORE_FACTORIES)


def _is_gql_init(val):
    n = val
    while n is not None and n.type in ("parenthesized_expression", "as_expression", "satisfies_expression"):
        n = n.named_children[0] if n.named_children else None
    if n is None or n.type != "call_expression":
        return False
    fn = n.child_by_field_name("function")
    args = n.child_by_field_name("arguments")
    return (fn is not None and fn.type == "identifier" and _txt(fn) in _GQL_TAGS and args is not None
            and (args.type == "template_string"
                 or any(c.type in ("template_string", "string") for c in args.children)))


def _mock_specifiers(root):
    for n in _iter(root):
        if n.type != "call_expression":
            continue
        fn = n.child_by_field_name("function")
        if fn is None or fn.type != "member_expression":
            continue
        obj = fn.child_by_field_name("object")
        prop = fn.child_by_field_name("property")
        if obj is None or prop is None or _txt(obj) not in _MOCK_OBJS or _txt(prop) not in _MOCK_METHODS:
            continue
        args = n.child_by_field_name("arguments")
        s = next((c for c in (args.children if args else []) if c.type == "string"), None)
        v = _str_value(s)
        if v:
            yield n.start_point[0] + 1, v


def _inline_gql_ops(root):
    """Named operations in gql templates that are not bound to a module-level const."""
    out = []
    for n in _iter(root):
        if n.type != "call_expression":
            continue
        fn = n.child_by_field_name("function")
        if fn is None or fn.type != "identifier" or _txt(fn) not in _GQL_TAGS:
            continue
        args = n.child_by_field_name("arguments")
        tmpl = args if (args is not None and args.type == "template_string") else next(
            (c for c in (args.children if args else []) if c.type == "template_string"), None)
        if tmpl is None:
            continue
        p = n.parent
        bound = p is not None and p.type == "variable_declarator" and (
            p.parent is not None and p.parent.parent is not None
            and (p.parent.parent.type == "program"
                 or (p.parent.parent.type == "export_statement" and p.parent.parent.parent is not None
                     and p.parent.parent.parent.type == "program")))
        if bound:
            continue
        m = _GQL_OP_RE.search(_txt(tmpl))
        if m:
            out.append((n.start_point[0] + 1, m.group(1), None))
    return out


def _tag_parts(name_node):
    """JSX/member tag -> (root identifier, [member path])."""
    if name_node is None:
        return None, []
    if name_node.type in ("identifier", "type_identifier"):
        return _txt(name_node), []
    if name_node.type in ("member_expression", "nested_identifier", "jsx_namespace_name"):
        parts = _txt(name_node).replace(":", ".").split(".")
        return parts[0], parts[1:]
    return None, []


def _on_jsx(node, fi, enc, line, resolve_ident, emit, is_gql):
    name = node.child_by_field_name("name")
    root_name, members = _tag_parts(name)
    if not root_name or not (root_name[0].isupper()):
        return
    r = resolve_ident(fi, root_name, members[0] if members else None)
    if not r:
        return
    emit(enc, r[2], "renders", "jsx", fi, line)


def _on_call(node, fi, enc, line, fnode, resolve_ident, is_store, is_gql, emit):
    fn = node.child_by_field_name("function")
    if fn is None:
        return
    if fn.type == "identifier":
        name = _txt(fn)
        if _HOOK_RE.match(name):
            r = resolve_ident(fi, name)
            if r:
                frel, local, nid, via = r
                emit(enc, nid, "calls_hook", "hook_call", fi, line)
                if is_store(frel, local):
                    emit(enc, nid, "uses_store", "store_call", fi, line)
    elif fn.type == "member_expression":
        obj = fn.child_by_field_name("object")
        prop = fn.child_by_field_name("property")
        if obj is not None and prop is not None and obj.type == "identifier" and \
                _HOOK_RE.match(_txt(prop)):
            # NS.useThing() where NS is a namespace import: link to the namespace's file
            r = resolve_ident(fi, _txt(obj), _txt(prop))
            if r:
                emit(enc, r[2], "calls_hook", "hook_call", fi, line)
                if is_store(r[0], r[1]):
                    emit(enc, r[2], "uses_store", "store_call", fi, line)
    # route factories
    callee = _root_callee(node)
    if callee is not None and callee.type == "identifier" and _ROUTE_FACTORY_RE.match(_txt(callee)):
        # only handle at the outermost call of the chain
        p = node.parent
        if p is not None and p.type == "call_expression" and p.child_by_field_name("function") == node:
            return
        c = node
        while c is not None and c.type == "call_expression":
            args = c.child_by_field_name("arguments")
            for a in (args.children if args else []):
                if a.type == "object":
                    _route_props(a, fi, fnode, resolve_ident, emit)
            c = c.child_by_field_name("function")


def _route_props(obj, fi, fnode, resolve_ident, emit):
    for pr in obj.children:
        key, val = None, None
        if pr.type == "pair":
            k = pr.child_by_field_name("key")
            key, val = _txt(k), pr.child_by_field_name("value")
        elif pr.type == "shorthand_property_identifier":
            key, val = _txt(pr), pr
        if key not in _ROUTE_COMPONENT_KEYS or val is None:
            continue
        for ident, ln in _component_idents(val):
            r = resolve_ident(fi, ident)
            if r:
                emit(fnode, r[2], "route_renders", "route", fi, ln)


def _component_idents(val):
    """Identifiers referenced as the route component: `X`, `<X/>`, `() => <X/>`, `memo(X)`."""
    out = []
    if val.type in ("identifier", "shorthand_property_identifier"):
        out.append((_txt(val), val.start_point[0] + 1))
    elif val.type in ("jsx_self_closing_element", "jsx_element"):
        op = val if val.type == "jsx_self_closing_element" else next(
            (c for c in val.children if c.type == "jsx_opening_element"), None)
        if op is not None:
            root, _m = _tag_parts(op.child_by_field_name("name"))
            if root and root[0].isupper():
                out.append((root, op.start_point[0] + 1))
    elif val.type == "arrow_function":
        body = val.child_by_field_name("body")
        if body is not None:
            for c in _iter(body):
                if c.type in ("jsx_self_closing_element", "jsx_opening_element"):
                    root, _m = _tag_parts(c.child_by_field_name("name"))
                    if root and root[0].isupper():
                        out.append((root, c.start_point[0] + 1))
                    break
    elif val.type == "call_expression":
        args = val.child_by_field_name("arguments")
        for a in (args.children if args else []):
            if a.type == "identifier":
                out.append((_txt(a), a.start_point[0] + 1))
                break
    return out


def _on_route_object(node, fi, fnode, line, resolve_ident, emit):
    """Router config objects (`{ path: '/x', component: X }` / `{ index: true, element: <X/> }`)."""
    keys = {}
    for pr in node.children:
        if pr.type == "pair":
            keys[_txt(pr.child_by_field_name("key"))] = pr
    if "path" not in keys and "index" not in keys:
        return
    # skip objects already handled as arguments of a route factory call
    p = node.parent
    if p is not None and p.type == "arguments":
        gp = p.parent
        if gp is not None and gp.type == "call_expression":
            callee = _root_callee(gp)
            if callee is not None and callee.type == "identifier" and _ROUTE_FACTORY_RE.match(_txt(callee)):
                return
    _route_props(node, fi, fnode, resolve_ident, emit)


def _on_jsx_attr(node, fi, enc, line, resolve_ident, emit):
    """`<Route component={X} element={<X/>} />` -> route_renders."""
    nm = next((c for c in node.children if c.type == "property_identifier"), None)
    if nm is None or _txt(nm) not in ("component", "Component", "element"):
        return
    el = node.parent
    if el is None or el.type not in ("jsx_opening_element", "jsx_self_closing_element"):
        return
    root, _m = _tag_parts(el.child_by_field_name("name"))
    if not root or not root.endswith("Route"):
        return
    expr = next((c for c in node.children if c.type == "jsx_expression"), None)
    if expr is None or not expr.named_children:
        return
    for ident, ln in _component_idents(expr.named_children[0]):
        r = resolve_ident(fi, ident)
        if r:
            emit(enc, r[2], "route_renders", "route", fi, ln)


def _on_member(node, fi, enc, line, resolve_ident, is_store, is_gql, emit, import_target):
    """`useStore.getState()` / `.setState` / `.subscribe` style access of a store binding."""
    obj = node.child_by_field_name("object")
    if obj is None or obj.type != "identifier":
        return
    name = _txt(obj)
    r = resolve_ident(fi, name)
    if not r:
        return
    frel, local, nid, via = r
    if is_store(frel, local) and frel != fi.rel:
        emit(enc, nid, "uses_store", "store_member", fi, line)
    elif is_store(frel, local) and enc != nid:
        # a store used from elsewhere in its own module (outside its definition)
        emit(enc, nid, "uses_store", "store_member", fi, line)


def _on_ident(node, fi, enc, line, resolve_ident, is_gql, emit):
    """Reference to an imported / local gql operation constant (outside import statements)."""
    name = _txt(node)
    if not name or not fi.imports.get(name) and name not in fi.gql_consts:
        return
    p = node.parent
    if p is None or p.type in ("import_specifier", "import_clause", "namespace_import",
                               "export_specifier", "variable_declarator", "property_identifier"):
        if p is not None and p.type == "variable_declarator" and p.child_by_field_name("name") == node:
            return
        if p is not None and p.type != "variable_declarator":
            return
    if p.type == "pair" and p.child_by_field_name("key") == node:
        return
    r = resolve_ident(fi, name)
    if not r:
        return
    frel, local, nid, via = r
    if is_gql(frel, local) and via:
        emit(enc, nid, "uses_gql_operation", "gql_operation", fi, line)

