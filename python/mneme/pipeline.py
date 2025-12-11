from __future__ import annotations

import random
import re
import subprocess
import sys
from enum import IntEnum
from typing import Dict, List, Tuple, Union

from mneme.logging import logger

__all_passes__ = """Module passes:
  always-inline
  annotation2metadata
  attributor
  attributor-light
  called-value-propagation
  canonicalize-aliases
  cg-profile
  check-debugify
  constmerge
  coro-cleanup
  coro-early
  cross-dso-cfi
  deadargelim
  debugify
  dfsan
  dot-callgraph
  dxil-upgrade
  elim-avail-extern
  embed-bitcode
  extract-blocks
  expand-variadics
  forceattrs
  function-import
  globalopt
  globalsplit
  heterogeneous-debug-verify
  hipstdpar-interpose-alloc
  hipstdpar-select-accelerator-code
  hotcoldsplit
  inferattrs
  inliner-ml-advisor-release
  inliner-wrapper
  inliner-wrapper-no-mandatory-first
  insert-gcov-profiling
  instrorderfile
  instrprof
  internalize
  invalidate<all>
  iroutliner
  jmc-instrumenter
  lower-emutls
  lower-global-dtors
  lower-ifunc
  lowertypetests
  memprof-context-disambiguation
  memprof-module
  mergefunc
  metarenamer
  module-inline
  name-anon-globals
  no-op-module
  objc-arc-apelim
  openmp-opt
  openmp-opt-postlink
  partial-inliner
  pgo-icall-prom
  pgo-instr-gen
  pgo-instr-use
  poison-checking
  print
  print-callgraph
  print-callgraph-sccs
  print-ir-similarity
  print-lcg
  print-lcg-dot
  print-must-be-executed-contexts
  print-profile-summary
  print-stack-safety
  print<inline-advisor>
  print<module-debuginfo>
  pseudo-probe
  pseudo-probe-update
  recompute-globalsaa
  rel-lookup-table-converter
  rewrite-statepoints-for-gc
  rewrite-symbols
  rpo-function-attrs
  sample-profile
  sancov-module
  sanmd-module
  scc-oz-module-inliner
  shadow-stack-gc-lowering
  strip
  strip-dead-debug-info
  strip-dead-prototypes
  strip-debug-declare
  strip-nondebug
  strip-nonlinetable-debuginfo
  synthetic-counts-propagation
  trigger-crash
  trigger-verifier-error
  tsan-module
  verify
  view-callgraph
  wholeprogramdevirt
Module passes with params:
  asan<kernel>
  global-merge<group-by-use;ignore-single-use;max-offset=N;merge-const;merge-external;no-group-by-use;no-ignore-single-use;no-merge-const;no-merge-external;size-only>
  globaldce<in-lto-post-link>
  hwasan<kernel;recover>
  ipsccp<no-func-spec;func-spec>
  loop-extract<single>
  memprof-use<profile-filename=S>
  msan<recover;kernel;eager-checks;track-origins=N>
  print<structural-hash><detailed>
Module analyses:
  callgraph
  collector-metadata
  inline-advisor
  ir-similarity
  lcg
  module-summary
  no-op-module
  pass-instrumentation
  profile-summary
  stack-safety
  verify
  globals-aa
Module alias analyses:
  globals-aa
CGSCC passes:
  argpromotion
  attributor-cgscc
  attributor-light-cgscc
  invalidate<all>
  no-op-cgscc
  openmp-opt-cgscc
CGSCC passes with params:
  coro-split<reuse-storage>
  function-attrs<skip-non-recursive-function-attrs>
  inline<only-mandatory>
CGSCC analyses:
  no-op-cgscc
  fam-proxy
  pass-instrumentation
Function passes:
  aa-eval
  adce
  add-discriminators
  aggressive-instcombine
  alignment-from-assumptions
  annotation-remarks
  assume-builder
  assume-simplify
  bdce
  bounds-checking
  break-crit-edges
  callbrprepare
  callsite-splitting
  chr
  codegenprepare
  consthoist
  constraint-elimination
  coro-elide
  correlated-propagation
  count-visits
  dce
  declare-to-assign
  dfa-jump-threading
  div-rem-pairs
  dot-cfg
  dot-cfg-only
  dot-dom
  dot-dom-only
  dot-post-dom
  dot-post-dom-only
  dse
  dwarf-eh-prepare
  expand-large-div-rem
  expand-large-fp-convert
  expand-memcmp
  fix-irreducible
  flattencfg
  float2int
  gc-lowering
  guard-widening
  gvn-hoist
  gvn-sink
  helloworld
  indirectbr-expand
  infer-address-spaces
  infer-alignment
  inject-tli-mappings
  instcount
  instnamer
  instsimplify
  interleaved-access
  interleaved-load-combine
  invalidate<all>
  irce
  jump-threading
  kcfi
  lcssa
  libcalls-shrinkwrap
  lint
  load-store-vectorizer
  loop-data-prefetch
  loop-distribute
  loop-fusion
  loop-load-elim
  loop-simplify
  loop-sink
  loop-versioning
  lower-constant-intrinsics
  lower-expect
  lower-guard-intrinsic
  lower-widenable-condition
  loweratomic
  lowerinvoke
  lowerswitch
  make-guards-explicit
  mem2reg
  memcpyopt
  memprof
  mergeicmps
  mergereturn
  move-auto-init
  nary-reassociate
  newgvn
  no-op-function
  objc-arc
  objc-arc-contract
  objc-arc-expand
  pa-eval
  partially-inline-libcalls
  pgo-memop-opt
  place-safepoints
  print
  print-alias-sets
  print-cfg-sccs
  print-memderefs
  print-mustexecute
  print-predicateinfo
  print<access-info>
  print<assumptions>
  print<block-freq>
  print<branch-prob>
  print<cost-model>
  print<cycles>
  print<da>
  print<debug-ata>
  print<delinearization>
  print<demanded-bits>
  print<domfrontier>
  print<domtree>
  print<func-properties>
  print<inline-cost>
  print<inliner-size-estimator>
  print<lazy-value-info>
  print<loops>
  print<memoryssa-walker>
  print<phi-values>
  print<postdomtree>
  print<regions>
  print<scalar-evolution>
  print<stack-safety-local>
  print<uniformity>
  reassociate
  redundant-dbg-inst-elim
  reg2mem
  safe-stack
  scalarize-masked-mem-intrin
  scalarizer
  sccp
  select-optimize
  separate-const-offset-from-gep
  sink
  sjlj-eh-prepare
  slp-vectorizer
  slsr
  stack-protector
  strip-gc-relocates
  structurizecfg
  tailcallelim
  tlshoist
  transform-warning
  trigger-verifier-error
  tsan
  typepromotion
  unify-loop-exits
  vector-combine
  verify
  verify<domtree>
  verify<loops>
  verify<memoryssa>
  verify<regions>
  verify<safepoint-ir>
  verify<scalar-evolution>
  view-cfg
  view-cfg-only
  view-dom
  view-dom-only
  view-post-dom
  view-post-dom-only
  wasm-eh-prepare
Function passes with params:
  cfguard<check;dispatch>
  early-cse<memssa>
  ee-instrument<post-inline>
  function-simplification<O1;O2;O3;Os;Oz>
  gvn<no-pre;pre;no-load-pre;load-pre;no-split-backedge-load-pre;split-backedge-load-pre;no-memdep;memdep>
  hardware-loops<force-hardware-loops;force-hardware-loop-phi;force-nested-hardware-loop;force-hardware-loop-guard;hardware-loop-decrement=N;hardware-loop-counter-bitwidth=N>
  instcombine<no-use-loop-info;use-loop-info;no-verify-fixpoint;verify-fixpoint;max-iterations=N>
  loop-unroll<O0;O1;O2;O3;full-unroll-max=N;no-partial;partial;no-peeling;peeling;no-profile-peeling;profile-peeling;no-runtime;runtime;no-upperbound;upperbound>
  loop-vectorize<no-interleave-forced-only;interleave-forced-only;no-vectorize-forced-only;vectorize-forced-only>
  lower-matrix-intrinsics<minimal>
  mldst-motion<no-split-footer-bb;split-footer-bb>
  print<da><normalized-results>
  print<memoryssa><no-ensure-optimized-uses>
  print<stack-lifetime><may;must>
  separate-const-offset-from-gep<lower-gep>
  simplifycfg<no-forward-switch-cond;forward-switch-cond;no-switch-range-to-icmp;switch-range-to-icmp;no-switch-to-lookup;switch-to-lookup;no-keep-loops;keep-loops;no-hoist-common-insts;hoist-common-insts;no-sink-common-insts;sink-common-insts;bonus-inst-threshold=N>
  speculative-execution<only-if-divergent-target>
  sroa<preserve-cfg;modify-cfg>
  win-eh-prepare<demote-catchswitch-only>
Function analyses:
  aa
  access-info
  assumptions
  bb-sections-profile-reader
  block-freq
  branch-prob
  cycles
  da
  debug-ata
  demanded-bits
  domfrontier
  domtree
  func-properties
  gc-function
  inliner-size-estimator
  lazy-value-info
  loops
  memdep
  memoryssa
  no-op-function
  opt-remark-emit
  pass-instrumentation
  phi-values
  postdomtree
  regions
  scalar-evolution
  should-not-run-function-passes
  should-run-extra-vector-passes
  ssp-layout
  stack-safety-local
  targetir
  targetlibinfo
  uniformity
  verify
  basic-aa
  objc-arc-aa
  scev-aa
  scoped-noalias-aa
  tbaa
Function alias analyses:
  basic-aa
  objc-arc-aa
  scev-aa
  scoped-noalias-aa
  tbaa
LoopNest passes:
  loop-flatten
  loop-interchange
  loop-unroll-and-jam
  no-op-loopnest
Loop passes:
  canon-freeze
  dot-ddg
  guard-widening
  indvars
  invalidate<all>
  loop-bound-split
  loop-deletion
  loop-idiom
  loop-instsimplify
  loop-predication
  loop-reduce
  loop-reroll
  loop-simplifycfg
  loop-unroll-full
  loop-versioning-licm
  no-op-loop
  print
  print<ddg>
  print<iv-users>
  print<loop-cache-cost>
  print<loopnest>
Loop passes with params:
  licm<allowspeculation>
  lnicm<allowspeculation>
  loop-rotate<no-header-duplication;header-duplication;no-prepare-for-lto;prepare-for-lto>
  simple-loop-unswitch<nontrivial;no-nontrivial;trivial;no-trivial>
Loop analyses:
  ddg
  iv-users
  no-op-loop
  pass-instrumentation
Machine module passes (WIP):
Machine function passes (WIP):
Machine function analyses (WIP):
  pass-instrumentation
"""


def split_top_level(s: str) -> list[str]:
    """
    Split on commas that are not nested inside parentheses.
    """
    parts = []
    buf = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def flatten_passes(s: str) -> list[str]:
    """
    Recursively extract only the *leaf* pass names from a pipeline string.
    Any pass that has parentheses is expanded, its *contents* flattened,
    and the outer name dropped.
    """
    out = []
    for tok in split_top_level(s):
        # if it has a parenthesized sub-list, peel it off
        m = re.match(r"^[^(]+\((.*)\)$", tok)
        if m:
            inner = m.group(1)
            # recurse into the contents
            out.extend(flatten_passes(inner))
        else:
            out.append(tok)
    return out


def test_pass_requires_analysis(pipeline_str, ir_path, output):
    result = subprocess.run(
        [
            "/opt/rocm-6.3.1/llvm/bin/opt",
            f"--passes={pipeline_str}",
            ir_path,
            "-o",
            output,
        ],
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
    )
    stderr = result.stderr.decode()

    return result.returncode, stderr


class PassLevel(IntEnum):
    MODULE = 0
    CGSCC = 1
    FUNCTION = 2
    LOOPNEST = 3
    # MSSA is not an adaptor, yet some passes require it so we wrap them around this adaptor
    LOOPMSSA = 4
    LOOP = 5
    UNKNOWN = 6


class PassOption:
    class OptionType(IntEnum):
        TOGGLE = 1
        RANGE = 2
        SETTING = 3

    def __init__(
        self,
        option: Union[str, List[str]],
        type: PassOption.OptionType,
        bound: int = 1,
    ):
        if isinstance(option, list):
            self.options = option
        else:
            self.option_name = option
        self._type = type
        self._bound = bound

    def is_toggle(self):
        return self._type == PassOption.OptionType.TOGGLE

    def is_setting(self):
        return self._type == PassOption.OptionType.SETTING

    def is_range(self):
        return self._type == PassOption.OptionType.RANGE

    def get_upper_bound(self):
        return self._bound


class AbstractPass:
    class ConcretePass:
        def __init__(self, apass: AbstractPass, options=None):
            self._apass = apass

            if options is not None:
                self.options = options
                return

            self.options = []
            if apass.options is None:
                return
            for k, v in apass.options.items():
                if v.is_toggle():
                    if random.choice([True, False]):
                        self.options.append(v.option_name)
                    else:
                        self.options.append("no-" + v.option_name)

                if v.is_range():
                    value = random.choice(list(range(1, v.get_upper_bound() + 1)))
                    self.options.append(v.option_name + "=" + str(value))

                if v.is_setting():
                    value = random.choice(list(range(0, len(v.options))))
                    self.options.append(v.options[value])

        def __str__(self):
            if len(self.options) == 0:
                return self._apass.pass_name

            pass_str = f"{self._apass.pass_name}" + "<" + ";".join(self.options) + ">"
            return pass_str

        def level(self):
            return self._apass.level

    def __init__(
        self,
        pass_name: str,
        options: Dict[str, PassOption],
        level: PassLevel,
        is_analysis: bool,
    ):
        self.pass_name = pass_name
        self.options = options
        self.level = level
        if "licm" == pass_name or "lnicm" == pass_name:
            self.level = PassLevel.LOOPMSSA

        self.is_analysis = is_analysis

    def __str__(self):
        if self.options is not None:
            opt = ",".join(
                [v[0] if v[1] is None else f"{v[0]}={v[1]}" for v in self.options]
            )
            return f"{self.pass_name}<{opt}>"
        else:
            return self.pass_name

    def optuna_pass(self, trial, identifier):
        options = []
        if len(options) == 0:
            return AbstractPass.ConcretePass(self)

        for k, v in self.options.items():
            if v.is_toggle():
                opt = trial.suggest_categorical(
                    f"{identifier}_{v.option_name}",
                    [v.option_name, "no-" + v.option_name],
                )
                options.append(opt)
            if v.is_range():
                opt = trial.suggest_int(
                    f"{identifier}_{v.option_name}", 1, v.get_upper_bound()
                )
                options.append(v.option_name + "=" + str(opt))
            if v.is_setting():
                for opt in v.options:
                    opt = trial.suggest_categorical(f"{identifier}_{v}", [True, False])
                    if opt is True:
                        options.append(v)
        return AbstractPass.ConcretePass(self, options)


class PipelineManager:
    @staticmethod
    def __parse_llvm_pipeline__(text: str) -> List[AbstractPass]:
        """
        Parse AVAIL_PASSES_TEXT into three categories of generators: module, cgscc, function.
        Sections are delimited by lines starting with '# Module passes', '# CGSCC passes',
        and '# Function passes'. Passes not under CGSCC or Function are treated as module.
        Returns a dict: {'module': [...], 'cgscc': [...], 'function': [...]}.
        """
        # ID : (HasOptions, Skip, Level}
        identifiers: Dict[str, Tuple[bool, bool, PassLevel]] = {
            "Module passes:": (False, False, PassLevel.MODULE),
            "Module passes with params:": (True, False, PassLevel.MODULE),
            "Module analyses:": (False, True, PassLevel.MODULE),
            "Module alias analyses:": (False, True, PassLevel.MODULE),
            "CGSCC passes:": (False, False, PassLevel.CGSCC),
            "CGSCC passes with params:": (True, False, PassLevel.CGSCC),
            "CGSCC analyses:": (False, True, PassLevel.CGSCC),
            "Function passes:": (False, False, PassLevel.FUNCTION),
            "Function passes with params:": (True, False, PassLevel.FUNCTION),
            "Function analyses:": (False, True, PassLevel.FUNCTION),
            "Function alias analyses": (False, True, PassLevel.FUNCTION),
            "LoopNest passes:": (
                False,
                True,
                PassLevel.LOOPNEST,
            ),  # We skip LoopNest as parser does not recognize this
            "Loop passes:": (False, False, PassLevel.LOOP),
            "Loop passes with params:": (True, False, PassLevel.LOOP),
            "Loop analyses:": (False, True, PassLevel.LOOP),
            "Machine module passes (WIP):": (False, True, PassLevel.UNKNOWN),
            "Machine function passes (WIP):": (False, True, PassLevel.UNKNOWN),
            "Machine function analyses (WIP):": (False, True, PassLevel.UNKNOWN),
        }
        current_state = None
        passes = list()
        skip = {
            "instrorderfile",
            "rewrite-statepoints-for-gc",
            "guard-widening",
            "reg2mem",
            "poison-checking",
            "print",
            "verify",
            "pseudo-probe",
            "annotation-remarks",
            "view",
            "hardware",
            "helloworld",
            "core",
            "bounds-checking",
            "attributor",
            "debugify",
            "debug",
            "select",
            "embed",
            "dot",
            "invalidate<all>",
            "pa-eval",
            "attributor-cgscc",
            "coro",
            "objc",
            "callbrprepare",
            "gc-lowering",
            "pgo",
            "verifier",
            "tsan",
            "profile",
            "canonicalize-aliases",
            "lint",
            "forceattrs",
            "msan",
            "metarenamer",
            "sancov-module",
            "asan",
            "insert-gcov-profiling",
            "lowertypetests",
            "instrprof",
            "synthetic‐counts‐propagation",
            "dfsan",
            "hwasan",
            "function-import",
            "inliner-ml-advisor-release",
            "sample-profile",
            "trigger-crash",
            "memprof",
            "codegenprepare",
            "verify",
            "aa-eval",
            "internalize",
            "extract-blocks",
            "strip-gc-relocates",
            "ee-instrument",
            "sancov-module",
            "globalsplit",
            "ipsccp<func-spec>",
            "openmp-opt-postlink",
            "strip",
            "newgvn",
            "no-op-cgscc",
            "no-op-function",
            "scc-oz-module-inliner",
            "structurizecfg",
            "iroutliner",
            "dwarf-eh-prepare",
            "jmc-instrumenter",
            "dxil-upgrade",
            "cross-dso-cfi",
            "sjlj-eh-prepare",
            "place-safepoints",
            "sanmd-module",
            "win-eh-prepare",
            "cfguard",
            "stack-protector",
            "count-visits",
            "lower-emutls",
            "hipstdpar-interpose-alloc",
            "safe-stack",
            "synthetic-counts-propagation",
            "add-discriminators",
            "transform-warning",
            "instnamer",
            "lower-guard-intrinsic",
            "kcfi",
            "wasm-eh-prepare",
            "instcount",
            "make-guards-explicit",
            "lower-ifunc",
            "tlshoist",
            "lower-global-dtors",
            "hotcoldsplit",
            "correlated-propagation",
            "speculative-execution",
            "chr",
            "assume-simplify",
            "assume-builder",
            "lowerinvoke",
            "libcalls-shrinkwrap",
            "mldst-motion",
            "interleaved-load-combine",
            "load-store-vectorizer",
            "inject-tli-mappings",
        }
        for line in text.splitlines():
            line = line.strip()
            if line in identifiers:
                current_state = line
                continue
            if current_state is None:
                raise RuntimeError("I cannot detect current state")

            options = identifiers.get(current_state, None)

            if options is None:
                raise RuntimeError("Cannot read option")

            # We skip all analysis
            if options[1]:
                continue
            skipflag = False
            for v in skip:
                if v in line:
                    skipflag = True
            if skipflag:
                continue

            if options[0]:
                name, params = parse_llvm_pass_options(line)
                passes.append(AbstractPass(name, params, options[2], options[1]))
            else:
                passes.append(AbstractPass(line, None, options[2], options[1]))

        return passes

    def __init__(self):
        self._passes = PipelineManager.__parse_llvm_pipeline__(__all_passes__)
        self._kw_passes = {k.pass_name: k for k in self._passes}
        self._kw_concrete_passes = {
            k.pass_name: AbstractPass.ConcretePass(k) for k in self._passes
        }
        logger.debug(f"Total LLVM exposed pipeline passes are: {len(self._passes)}")

    def get_passes(self) -> List[str]:
        return list(self._kw_concrete_passes.keys())

    def get_concrete_passes(self) -> Dict[str, AbstractPass.ConcretePass]:
        return self._kw_concrete_passes

    def split_pipeline(self, pipeline):
        return __flatten_passes__(__split_top_level__(pipeline))

    def _generate_random_pipeline(
        self,
        mean_passes: float,
        std_passes: float,
        allow_repeats: bool = False,
    ) -> List[AbstractPass.ConcretePass]:
        """
        Generate `samples` random LLVM-pass pipelines.

        Each pipeline length is drawn from N(mean_passes, std_passes) and then
        clamped to [1, len(passes_list)]. Passes are sampled from passes_list
        either with or without replacement.

        Args:
          passes_list:        your list of available pass names.
          samples:            how many pipelines to generate.
          mean_passes:        target average number of passes per pipeline.
          std_passes:         desired standard deviation of pipeline lengths.
          allow_repeats:      if True, pipelines may include the same pass multiple times.

        Returns:
          A list of `samples` pipelines, each a list of pass names.
        """

        max_len = len(self._passes)

        raw = random.gauss(mean_passes, std_passes)
        length = int(round(raw))
        length = max(1, min(max_len, length))

        # 3) pick that many passes
        pipeline = []
        if allow_repeats:
            pipeline = [
                AbstractPass.ConcretePass(random.choice(self._passes))
                for _ in range(length)
            ]
        else:
            pipeline = [
                AbstractPass.ConcretePass(v)
                for v in random.sample(self._passes, length)
            ]

        return pipeline

    def generate(self, samples, mean_passes, std, allow_repeats=True, seed=0):
        random.seed(seed)
        random_selection = []

        for _ in range(samples):
            sp = self._generate_random_pipeline(mean_passes, std, allow_repeats)
            random_selection.append(sp)

        return random_selection

    def from_string(self, str_pipeline: str) -> List[AbstractPass.ConcretePass]:
        def get_options(names):
            if len(names) <= 1:
                return []

            opts_str = names[1:]
            opts_str[-1] = opts_str[-1].rstrip(">")
            options = (
                [opts_str[-1].strip() for opt in opts_str[-1].split(",")]
                if opts_str
                else []
            )
            return options

        # We first replace all adaptors.
        no_funtion = str_pipeline.replace("function<eager-inv>(", "")
        no_funtion = no_funtion.replace("function(", "")
        no_cgscc = no_funtion.replace("cgscc(", "")
        no_loop = no_cgscc.replace("loop(", "")
        no_loopmassa = no_loop.replace("loop-mssa(", "")
        no_loopnest = no_loopmassa.replace("loopnest", "")
        passes = no_loopnest.replace(")", "").split(",")

        passes = [x for x in passes if x != ""]
        pipeline = []
        for p in passes:
            names = p.split("<", 1)
            name = names[0]
            options = get_options(names)
            if name not in self._kw_passes:
                raise RuntimeError(f"Unknown pass {name}")
            pipeline.append(AbstractPass.ConcretePass(self._kw_passes[name], options))
        return pipeline

    @staticmethod
    def to_string(passes: List[AbstractPass.ConcretePass]) -> str:
        currentLevel = PassLevel.MODULE
        nesting = [currentLevel]

        _pipeline = ""
        for p in passes:
            if p.level() == currentLevel:
                _pipeline += f",{p}"
            elif currentLevel == PassLevel.MODULE:
                if p.level() == PassLevel.FUNCTION:
                    _pipeline += f",function<eager-inv>({p}"
                    currentLevel = PassLevel.FUNCTION
                elif p.level() == PassLevel.CGSCC:
                    _pipeline += f",cgscc({p}"
                    currentLevel = PassLevel.CGSCC
                elif p.level() == PassLevel.LOOPNEST:
                    _pipeline += f",function<eager-inv>(loopnest({p}"
                    currentLevel = PassLevel.LOOPNEST
                elif p.level() == PassLevel.LOOP:
                    _pipeline += f",function<eager-inv>(loop({p}"
                    currentLevel = PassLevel.LOOP
                elif p.level() == PassLevel.LOOPMSSA:
                    _pipeline += f",function<eager-inv>(loop-mssa({p}"
                    currentLevel = PassLevel.LOOPMSSA
                else:
                    raise RuntimeError(f"Unknown level {p.level()}")
            elif currentLevel == PassLevel.FUNCTION:
                if p.level() == PassLevel.MODULE:
                    _pipeline += f"),{p}"
                    currentLevel = PassLevel.MODULE
                elif p.level() == PassLevel.CGSCC:
                    _pipeline += f"),cgscc({p}"
                    currentLevel = PassLevel.CGSCC
                elif p.level() == PassLevel.LOOPNEST:
                    _pipeline += f",loopnest({p}"
                    currentLevel = PassLevel.LOOPNEST
                elif p.level() == PassLevel.LOOP:
                    _pipeline += f",loop({p}"
                    currentLevel = PassLevel.LOOP
                elif p.level() == PassLevel.LOOPMSSA:
                    _pipeline += f",loop-mssa({p}"
                    currentLevel = PassLevel.LOOPMSSA
                else:
                    raise RuntimeError(f"Unknown level {p.level()} {p}")
            elif currentLevel == PassLevel.CGSCC:
                if p.level() == PassLevel.MODULE:
                    _pipeline += f"),{p}"
                    currentLevel = PassLevel.MODULE
                elif p.level() == PassLevel.FUNCTION:
                    _pipeline += f"),function<eager-inv>({p}"
                    currentLevel = PassLevel.FUNCTION
                elif p.level() == PassLevel.LOOPNEST:
                    _pipeline += f"),function<eager-inv>(loopnest({p}"
                    currentLevel = PassLevel.LOOPNEST
                elif p.level() == PassLevel.LOOP:
                    _pipeline += f"),function<eager-inv>(loop({p}"
                    currentLevel = PassLevel.LOOP
                elif p.level() == PassLevel.LOOPMSSA:
                    _pipeline += f"),function<eager-inv>(loop-mssa({p}"
                    currentLevel = PassLevel.LOOPMSSA
                else:
                    raise RuntimeError(f"Unknown level {p.level()}")
            elif currentLevel == PassLevel.LOOP:
                if p.level() == PassLevel.MODULE:
                    _pipeline += f")),{p}"
                    currentLevel = PassLevel.MODULE
                elif p.level() == PassLevel.CGSCC:
                    _pipeline += f")),cgscc({p}"
                    currentLevel = PassLevel.CGSCC
                elif p.level() == PassLevel.FUNCTION:
                    _pipeline += f"),{p}"
                    currentLevel = PassLevel.FUNCTION
                elif p.level() == PassLevel.LOOPNEST:
                    _pipeline += f"),loopnest({p}"
                    currentLevel = PassLevel.LOOPNEST
                elif p.level() == PassLevel.LOOPMSSA:
                    _pipeline += f"),loop-mssa({p}"
                    currentLevel = PassLevel.LOOPMSSA
                else:
                    raise RuntimeError(f"Unknown level {p.level()}")
            elif currentLevel == PassLevel.LOOPMSSA:
                if p.level() == PassLevel.MODULE:
                    _pipeline += f")),{p}"
                    currentLevel = PassLevel.MODULE
                elif p.level() == PassLevel.FUNCTION:
                    _pipeline += f")),function({p}"
                    currentLevel = PassLevel.FUNCTION
                elif p.level() == PassLevel.CGSCC:
                    _pipeline += f")),cgscc({p}"
                    currentLevel = PassLevel.CGSCC
                elif p.level() == PassLevel.LOOPNEST:
                    _pipeline += f"),loopnest({p}"
                    currentLevel = PassLevel.LOOPNEST
                elif p.level() == PassLevel.LOOP:
                    _pipeline += f"),loop({p}"
                    currentLevel = PassLevel.LOOP
                else:
                    raise RuntimeError(f"Unknown level {p.level()} {p}")
            elif currentLevel == PassLevel.LOOPNEST:
                if p.level() == PassLevel.MODULE:
                    _pipeline += f")),{p}"
                    currentLevel = PassLevel.MODULE
                elif p.level() == PassLevel.CGSCC:
                    _pipeline += f")),cgscc({p}"
                    currentLevel = PassLevel.CGSCC
                elif p.level() == PassLevel.FUNCTION:
                    _pipeline += f"),{p}"
                    currentLevel = PassLevel.FUNCTION
                elif p.level() == PassLevel.LOOP:
                    _pipeline += f"),loop({p}"
                    currentLevel = PassLevel.LOOP
                elif p.level() == PassLevel.LOOPMSSA:
                    _pipeline += f"),loop-mssa({p}"
                    currentLevel = PassLevel.LOOPMSSA
                else:
                    raise RuntimeError(f"Unknown level {p.level()}")
            else:
                raise RuntimeError(f"Unsupported case")

        if currentLevel == PassLevel.FUNCTION or currentLevel == PassLevel.CGSCC:
            _pipeline += ")"
        elif (
            currentLevel == PassLevel.LOOP
            or currentLevel == PassLevel.LOOPNEST
            or currentLevel == PassLevel.LOOPMSSA
        ):
            _pipeline += "))"
        return _pipeline[1:]


def parse_llvm_pass_options(llvm_pass):
    m = re.match(r"^([^<\s]+)((?:<[^>]+>)*)$", llvm_pass)
    if not m:
        raise RuntimeError(
            "I am expecing pass to have options, yet I cannot detect any"
        )
    name, group_str = m.groups()
    raw_groups = re.findall(r"<([^>]+)>", group_str)
    pass_options = {}
    uncategorized = []
    for group in raw_groups:
        options = group.split(";")
        for opt in options:
            if "lto" in opt:
                continue
            if "=" in opt:
                kv = opt.split("=")
                if len(kv) != 2:
                    raise RuntimeError(f"Option {opt} has unrecognized structure {kv}")
                if kv[1] == "N":
                    pass_options[kv[0]] = PassOption(
                        kv[0], PassOption.OptionType.RANGE, 1  # 32
                    )
            else:
                uncategorized.append(opt)

        for uc in uncategorized:
            base = uc[3:] if uc.startswith("no-") else uc
            if base in pass_options:
                continue

            toggle = "no-" + base
            if toggle in uncategorized and base in uncategorized:
                pass_options[base] = PassOption(base, PassOption.OptionType.TOGGLE)

        values = []
        for uc in uncategorized:
            base = uc[3:] if uc.startswith("no-") else uc
            if base in pass_options:
                continue
            values.append(uc)

        if len(values) > 0:
            pass_options[name] = PassOption(values, PassOption.OptionType.SETTING)

    return name, pass_options
