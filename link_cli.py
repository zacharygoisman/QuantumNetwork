#link_cli.py
"""
Link_CLI: a minimal interactive prompt for editing a built topology.

The default ``add``/``remove`` verbs operate on **link pairs** -- the
entangled user pairs that the pipeline routes (the *colored* lines in
``network_plot.svg``). To edit the underlying physical fiber graph
(the gray edges) use ``add_edge`` / ``remove_edge``.

Supported commands (exactly one per line):

    add <u> <v>            -- add an entangled user pair to route
    add <u> <v> <fid>      -- add a pair with an explicit fidelity threshold
    remove <u> <v>         -- remove an entangled user pair
    save <path>            -- save the current topology to a JSON snapshot
    load <path>            -- replace the current topology from a JSON snapshot
    combos <n>             -- override cfg.max_combos for subsequent runs
    run                    -- re-run the pipeline against the current topology
    quit                   -- exit the prompt

Errors raised by the Link_Editor (`ValueError`, `OSError`) are printed and
the prompt loop continues.

Run from the project root with:

    python -m link_cli
    # or
    python link_cli.py [preset_name]
"""

#ZHG
#2026.07.19
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import shlex
import sys

from config.presets import (
    contention,
    manhattan_ilec,
    paper_dense,
    paper_exhaustive,
    paper_ring,
    simple_contention,
    star,
    super_dense,
    two_source_three_users_custom, paper_dense_copy
)
from config.custom_topology import my_custom_network
from network.builder import build_network
from network.link_editor import (
    add_link_pair,
    load_topology,
    remove_link_pair,
    save_topology,
)
from pipeline.runner import run_pipeline_prebuilt


PROMPT = "topology> "

# ``verb -> (min_args, max_args)``. Every verb defined here is dispatched
# by ``LinkCLI._dispatch``; any first token not in this map prints usage.
_ARG_COUNTS = {
    "add": (2, 3),
    "remove": (2, 2),
    "save": (1, 1),
    "load": (1, 1),
    "combos": (1, 1),
    "run": (0, 0),
    "quit": (0, 0),
}


def _usage_line():
    """Return the single-line usage summary printed on invalid input."""
    parts = []
    for verb, (lo, hi) in _ARG_COUNTS.items():
        if lo == hi:
            parts.append(f"{verb}({lo})")
        else:
            parts.append(f"{verb}({lo}-{hi})")
    return "usage: " + ", ".join(parts)


def _rebuild_sources_from_topology(topology, cfg):
    """Rebuild the ``sources`` dict for a (possibly loaded) topology.

    We re-derive source channel lists from ``cfg`` -- either the explicit
    ``cfg.custom_source_channels`` mapping or the positional
    ``cfg.num_channels`` list -- and only include labels that actually
    exist as ``node_type == 'source'`` in the topology. This lets the
    CLI keep operating after ``load`` even when the loaded snapshot has
    a different set of nodes than the original ``build_network`` output.
    """
    source_labels = sorted(
        n for n, data in topology.nodes(data=True)
        if data.get("node_type") == "source"
    )

    sources = {}
    if getattr(cfg, "custom_source_channels", None):
        for s in source_labels:
            if s in cfg.custom_source_channels:
                sources[s] = {
                    "available_channels": list(cfg.custom_source_channels[s])
                }
        return sources

    num_channels = list(cfg.num_channels or [])
    for i, s in enumerate(source_labels):
        # Fall back to the last configured channel count if we have more
        # source nodes than cfg entries (e.g. after loading a bigger
        # snapshot). Better a working default than an IndexError.
        if i < len(num_channels):
            n = int(num_channels[i])
        elif num_channels:
            n = int(num_channels[-1])
        else:
            n = 1
        sources[s] = {"available_channels": list(range(1, n + 1))}
    return sources


_WELCOME_BANNER = """\
=====================================================================
Link_CLI  --  interactive editor for entangled user pairs
=====================================================================

Available commands (one per line):
  add <u> <v>            add an entangled user pair (colored link)
  add <u> <v> <fid>      same, with an explicit fidelity target (0.5-1.0)
  remove <u> <v>         remove an entangled user pair
  save <path>            save the current topology to a JSON snapshot
  load <path>            replace the current topology from a JSON snapshot
  combos <n>             cap combos per run (or 'none' for unlimited)
  run                    re-run the pipeline against the current topology
  quit                   exit the prompt

How to use:
  1. Type `add U? U?` to introduce new link demands.
  2. Type `remove U? U?` to drop an existing pair.
  3. Type `run` when you want to route + allocate the current setup.
  4. After each `run`, open outputs/network_plot.svg to see the updated
     routing (colored paths). Other artifacts land in the same folder.
====================================================================="""


class LinkCLI:
    """Interactive prompt state.

    Kept as a class so that ``quit`` / EOF cleanly exits by returning from
    :meth:`run` without having to thread a flag through helper functions.
    """

    def __init__(self, cfg, *, input_stream=None, output_stream=None):
        self.cfg = cfg
        self.input = input_stream if input_stream is not None else sys.stdin
        self.output = output_stream if output_stream is not None else sys.stdout

        # Initial state: build once, then let the user mutate the graph.
        self.topology, self.sources, self.links = build_network(cfg)

    # ---------------------------------------------------------------- IO

    def _print(self, message):
        print(message, file=self.output, flush=True)

    def _prompt(self):
        # ``input()`` reads from stdin only; use the streams we were given
        # so tests can drive the CLI with StringIO.
        self.output.write(PROMPT)
        self.output.flush()
        return self.input.readline()

    # ------------------------------------------------------------ verbs
    #
    # add / remove operate on entangled user pairs (the colored lines in
    # network_plot.svg). Physical fiber connections (the gray lines) are
    # fixed by the preset and can only be changed by editing the config
    # or by loading a different topology snapshot.

    def _cmd_add(self, args):
        u, v = args[0], args[1]
        fidelity = args[2] if len(args) == 3 else None
        pair = add_link_pair(
            self.topology, self.links, u, v,
            cfg=self.cfg, fidelity=fidelity,
        )
        # Report the resulting fidelity too, since it drives allocation.
        f_used = self.cfg.fidelity_limit[-1] if getattr(self.cfg, "fidelity_limit", None) else None
        if f_used is not None:
            self._print(f"add {pair[0]} {pair[1]} fidelity={f_used}")
        else:
            self._print(f"add {pair[0]} {pair[1]}")

    def _cmd_remove(self, args):
        u, v = args[0], args[1]
        pair = remove_link_pair(self.topology, self.links, u, v, cfg=self.cfg)
        self._print(f"remove {pair[0]} {pair[1]}")

    def _cmd_save(self, args):
        path = args[0]
        save_topology(self.topology, path)
        self._print(f"save {path}")

    def _cmd_load(self, args):
        path = args[0]
        # Load-then-swap: if load raises, the in-memory topology is
        # untouched.
        new_topology = load_topology(path)
        self.topology = new_topology
        self.sources = _rebuild_sources_from_topology(self.topology, self.cfg)
        self._print(f"load {path}")

    def _cmd_combos(self, args):
        raw = args[0]
        if raw.lower() in ("none", "all", "unlimited"):
            self.cfg.max_combos = None
            self._print("combos unlimited")
            return
        try:
            n = int(raw)
        except ValueError:
            raise ValueError(f"combos: expected an integer or 'none', got {raw!r}")
        if n <= 0:
            raise ValueError(f"combos: value must be positive, got {n}")
        self.cfg.max_combos = n
        self._print(f"combos {n}")

    def _cmd_run(self, args):
        run_pipeline_prebuilt(self.cfg, self.topology, self.sources, self.links)
        # Point the user at the freshly-rendered plot so it's obvious where
        # to look after each run.
        outdir = getattr(self.cfg, "output_directory", "outputs")
        self._print(
            f"\nSee {outdir}/network_plot.svg for the updated routing"
        )

    # ----------------------------------------------------------- dispatch

    def _dispatch(self, tokens):
        verb = tokens[0]
        args = tokens[1:]

        if verb not in _ARG_COUNTS:
            self._print(_usage_line())
            return True

        lo, hi = _ARG_COUNTS[verb]
        if not (lo <= len(args) <= hi):
            self._print(_usage_line())
            return True

        if verb == "quit":
            return False

        try:
            if verb == "add":
                self._cmd_add(args)
            elif verb == "remove":
                self._cmd_remove(args)
            elif verb == "save":
                self._cmd_save(args)
            elif verb == "load":
                self._cmd_load(args)
            elif verb == "combos":
                self._cmd_combos(args)
            elif verb == "run":
                self._cmd_run(args)
        except (ValueError, OSError) as e:
            # Print the message, keep going, leave state unchanged
            # (link_editor guarantees the latter).
            self._print(str(e))

        return True

    # --------------------------------------------------------------- loop

    def run(self):
        """Run the prompt loop until the user quits or hits EOF."""
        self._print(_WELCOME_BANNER)
        while True:
            line = self._prompt()
            if line == "":
                # EOF (``readline`` returns "" -- a blank line would be "\n").
                return

            stripped = line.strip()
            if not stripped:
                # Blank / whitespace-only line: no action, keep prompting.
                continue

            try:
                tokens = shlex.split(stripped)
            except ValueError as e:
                # e.g. an unbalanced quote. Treat as a malformed command.
                self._print(f"parse error: {e}")
                continue

            if not tokens:
                continue

            if not self._dispatch(tokens):
                return


# --------------------------------------------------------------------------- #
# Presets available from the CLI's command-line launcher
# --------------------------------------------------------------------------- #

_PRESETS = {
    "contention": contention,
    "manhattan": manhattan_ilec,
    "paper_dense": paper_dense,
    "paper_exhaustive": paper_exhaustive,
    "paper_ring": paper_ring,
    "simple_contention": simple_contention,
    "star": star,
    "super_dense": super_dense,
    "two_source_three_users_custom": two_source_three_users_custom,
    "custom": my_custom_network,
    "paper_dense_copy": paper_dense_copy,
}


def _resolve_cfg(preset_name):
    if preset_name not in _PRESETS:
        raise SystemExit(
            f"unknown preset {preset_name!r}. "
            f"Available: {', '.join(sorted(_PRESETS))}"
        )
    return _PRESETS[preset_name]()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    preset = argv[0] if argv else "paper_dense_copy"
    cfg = _resolve_cfg(preset)
    LinkCLI(cfg).run()


if __name__ == "__main__":
    main()
