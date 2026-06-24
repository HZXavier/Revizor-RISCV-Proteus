# PPS2 — Securing Proteus

**Automated differential analysis of a RISC-V soft-core**

---
## Table of Contents

- [Overview](#overview)
- [What the framework detects](#what-the-framework-detects)
- [Why this project is novel](#why-this-project-is-novel)
- [How the project is structured](#how-the-project-is-structured)
- [At a glance](#at-a-glance)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Workflow](#workflow)
- [Historical PoCs](#historical-pocs)
- [Reproducing results](#reproducing-results)
- [Architecture summary](#architecture-summary)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

Modern processors use speculative execution to stay fast: they guess the
outcome of branches and run ahead before the guess is confirmed. When the
guess is wrong, the processor rolls back the architectural state — but **not
the cache**. This gap is what vulnerabilities like Spectre exploit: a secret
value briefly influences which memory address is touched, and even though the
result is discarded, the cache footprint survives and can be read through a
timing side-channel.

Hardware is unforgiving in this respect. Once a chip is manufactured you
cannot patch transistors — a microarchitectural flaw baked into silicon is
permanent. The only real opportunity to catch these flaws is **before
manufacturing**, while the design still exists as RTL (Register Transfer
Level) code that can be simulated.

PPS2 is a framework that does exactly this for **Proteus**, an open-source
RISC-V soft-core. It generates random RV32IM programs, runs them on two
independent executors, and compares the results:

- **Unicorn** — a pure software ISA emulator with no microarchitecture. It
  tells us what a program is *supposed* to compute (the ground truth).
- **Proteus** — a cycle-accurate out-of-order RTL simulator with a real
  pipeline, caches, and branch predictor. It tells us what the hardware
  *actually* does.

Any disagreement between the two is either a functional bug or, more
interestingly, an information leak — a place where the hardware reveals
secret-dependent behaviour that the ISA says should be invisible.

### What the framework detects

The analysis is built on **leakage contracts** (Guarnieri et al. 2021): a
formal way of stating what an attacker is allowed to observe. PPS2 implements
two contracts:

- **CT-SEQ** — *Constant-Time, Sequential.* Two runs of the same program with
  different secret inputs must access the same memory addresses. A difference
  means a secret leaks through the memory access pattern, even without
  speculation.
- **CT-COND** — *Constant-Time, Conditional speculation.* Same idea, but now
  including the instructions executed speculatively on a mispredicted branch.
  A violation here is **Spectre variant 1**.

### Why this project is novel

No existing tool combines all three of:

- an **open-source RISC-V** target,
- a **real out-of-order RTL** executor (not a behavioural model like gem5),
- **hardware-software leakage contracts**.

Revizor (Microsoft, 2022) is x86-only and depends on gem5. DifuzzRTL fuzzes
RISC-V RTL but finds functional bugs, not leaks. Scam-V targets ARM with no
out-of-order RTL. PPS2 fills that gap, with no dependency on Revizor or gem5.

---

## How the project is structured

The codebase was built incrementally, each script solving one concrete
problem before moving to the next. The scripts are best understood as a
**pipeline that grows in three phases**, mirroring how the project actually
developed.

### Phase 1 — Build a vocabulary of instructions

Before you can fuzz a processor you need a precise, machine-readable
description of the instructions you are allowed to generate. We could not
reuse Revizor's x86 XML format, so we built our own.

1. **`create_instructions.py`** — writes `riscv_isa.json`, a structured spec
   of every RV64IM instruction (operands, types, categories). This is the
   single source of truth for everything downstream.
2. **`filter_rv32.py`** — Proteus implements RV32, not RV64, so this strips
   the 64-bit-only instructions (`*W` variants, `LD`, `SD`, `LWU`) and fixes
   the shift amounts. Produces `riscv_isa_rv32.json` (45 instructions).
3. **`validate_isa.py`** — a structural sanity check: are all fields present,
   do immediate ranges fit their bit widths, is every control-flow
   instruction correctly tagged?
4. **`test_instructions.py`** — the practical check: can a real assembler
   actually assemble each instruction we generate? Catches malformed operands
   the structural validator can't see.

At the end of Phase 1 we have a trustworthy instruction set and a generator
that produces valid assembly.

### Phase 2 — Get programs to *run* somewhere

Generating assembly is useless if you can't execute it. The next problem was
getting a single instruction to run end-to-end and observe its effect. This
was solved iteratively, which is why there are three proof-of-concept scripts:

5. **`riscv_unicorn_poc.py`** *(PoC v1)* — proves Unicorn can execute one
   generated instruction at all.
6. **`riscv_unicorn_poc2.py`** *(PoC v2)* — adds operand lookup by name and a
   before/after register diff, so we can actually *see* what the instruction
   changed.
7. **`riscv_unicorn_loadstores_tb.py`** *(PoC v3)* — introduces the **memory
   sandbox**: a reserved register (`x30`) holding a valid base address, so
   loads and stores work without faulting. This sandbox design becomes the
   backbone of the entire production pipeline.

These three are kept as **historical witnesses**. They are not used in the
final pipeline, but they document the path from "nothing runs" to "memory
works".

### Phase 3 — Run on real hardware and compare

With a working sandbox, we could finally bring in Proteus and build the actual
analysis. This is the production pipeline.

8. **`proteus_runner.py`** *(Level 1)* — runs one instruction on the Proteus
   RTL simulator inside Docker, and reads back the result. This required
   modifying the simulator (`main.cpp`) to add three things: a fast halt
   mechanism, a register dump, and a memory-access trace.
9. **`diff_tester.py`** *(Level 2)* — the core idea of the project: compile
   *one* program, run it on **both** Unicorn and Proteus from an identical
   initial state, and compare all 25 tracked registers. 100/100 match
   confirms Proteus correctly implements RV32IM.
10. **`contract_checker.py`** *(Level 3a)* — moves from correctness to
    security. Runs the same program twice with different secret inputs and
    compares the **memory access traces** (CT-SEQ). A difference is a leak.
11. **`cond_checker.py`** *(Level 3b)* — the most advanced step. Generates
    Spectre-v1 gadgets and simulates branch misprediction (always-mispredict
    model), collecting the memory accesses made on the *wrong* speculative
    path. A divergence is an automatically-detected Spectre vulnerability.

### The supporting piece

- **`main.cpp`** — the modified Proteus simulator. It exposes the hardware to
  the Python scripts through three additions: the **CharDev EOT halt**
  (instant stop instead of 77 seconds), the **REGDUMP** block (register state
  after execution), and the **MEMTRACE** block (every memory access with its
  cycle). Without these, Proteus is a black box; with them, it becomes
  observable.

### The trame in one line

```
Define instructions  →  Run them somewhere  →  Run on real HW & compare
   (Phase 1)               (Phase 2, PoCs)         (Phase 3, Levels 1-3)
   spec + validate         sandbox emerges         correctness then security
```

---

## At a glance

| Level | What it does | Result |
|---|---|---|
| 1 — Proteus execution | Run RV32IM programs on Proteus RTL via Docker | Register state + memory trace |
| 2 — Differential testing | Compare Proteus vs Unicorn register state | 100/100 match on RV32IM |
| 3a — CT-SEQ | Detect data-dependent memory leaks | Violation detected on `lb` |
| 3b — CT-COND | Detect Spectre variant 1 gadgets | 8/30 gadgets violated |

---

## Repository layout

```
PPS2/
├── proteus/                        # Proteus RTL soft-core (submodule)
│   ├── sim/
│   │   └── main.cpp                # Modified simulator: MEMTRACE + REGDUMP + CharDev halt
│   └── Dockerfile
└── revizor-riscv/
    ├── venv/                       # Python virtual environment
    └── src/generator/
        ├── riscv_isa.json          # Generated RV64IM spec (62 instructions)
        ├── riscv_isa_rv32.json     # Filtered RV32IM spec (45 instructions)
        ├── create_instructions.py  # Generate riscv_isa.json
        ├── filter_rv32.py          # Filter RV64 → RV32IM
        ├── validate_isa.py         # Structural validator for JSON specs
        ├── test_instructions.py    # Assemble each instruction via riscv64-elf-as
        ├── riscv_unicorn_poc.py    # PoC v1: basic Unicorn execution (historical)
        ├── riscv_unicorn_poc2.py   # PoC v2: named operands + register diff (historical)
        ├── riscv_unicorn_loadstores_tb.py  # PoC v3: memory sandbox (historical)
        ├── proteus_runner.py       # Level 1: run one instruction on Proteus
        ├── diff_tester.py          # Level 2: differential testing Unicorn vs Proteus
        ├── contract_checker.py     # Level 3a: CT-SEQ leakage detection
        └── cond_checker.py         # Level 3b: CT-COND Spectre-v1 detection
```

---

## Prerequisites

### System dependencies

| Tool | Purpose | Install |
|---|---|---|
| Docker Desktop | Proteus simulator + RV32 cross-compiler | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Python 3.9+ | All Python scripts | Pre-installed on macOS |
| `riscv64-elf-as` | Local assembler for `test_instructions.py` | `brew install riscv-gnu-toolchain` |

> **macOS / Apple Silicon:** Docker Desktop supports ARM64 natively.
> The Dockerfile detects the host architecture automatically and downloads
> the correct Coursier binary.

### Python packages

All Python scripts use a shared virtual environment:

```
unicorn          # CPU emulator (RV32IM)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/HZXavier/Revizor-RISCV-Proteus.git PPS2
cd PPS2
git submodule update --init --recursive
```

### 2. Set up the Python virtual environment

```bash
cd revizor-riscv
python3 -m venv venv
source venv/bin/activate
pip install unicorn
```

### 3. Build the Docker image

The Docker image contains the RV32 cross-compiler and the Proteus simulator.
**This step takes 20–40 minutes** (toolchain compilation from source).

```bash
cd proteus
docker build -t proteus .
```

Verify the build:

```bash
docker run --rm proteus riscv32-unknown-elf-gcc --version
docker run --rm proteus /proteus/sim/build/sim 2>&1 | head -2
```

---

## Workflow

All scripts are in `revizor-riscv/src/generator/`.
Always activate the venv before running them:

```bash
cd revizor-riscv/src/generator
source ../../venv/bin/activate
```

---

### Step 1 — Generate the ISA specification

```bash
# Generate the RV64IM spec (62 instructions)
python3 create_instructions.py
# → riscv_isa.json

# Filter down to RV32IM (45 instructions, removes *W / LD / SD / LWU)
python3 filter_rv32.py
# → riscv_isa_rv32.json
```

Expected output:
```
[OK] 45/62 instructions kept (RV32IM) -> riscv_isa_rv32.json
  RV32I-ARITH          10
  RV32I-ARITH-IMM       6
  RV32I-COND-BR         6
  ...
```

### Step 2 — Validate and test the spec

```bash
# Structural validation
python3 validate_isa.py riscv_isa_rv32.json

# Assemble every instruction via riscv64-elf-as (requires local toolchain)
python3 test_instructions.py riscv_isa_rv32.json --all
```

---

### Step 3 — Level 1: run one instruction on Proteus

Requires Docker running.

```bash
python3 proteus_runner.py
python3 proteus_runner.py --seed 42
```

Expected output:
```
Instruction : SLTI
Assembly    : slti x22, x15, -1971
...
[OK] Proteus output:
Memory trace (25 accesses):
  write  0x80002000
  ...
Register state after execution:
  x5  = 0x00000001
  ...
Simulation done after 20 cycles.
```

---

### Step 4 — Level 2: differential testing (Unicorn vs Proteus)

Requires Docker running.

```bash
# One test case
python3 diff_tester.py --seed 42

# 100 test cases
python3 diff_tester.py --runs 100 --seed 42
```

Expected output:
```
Base seed : 42
Runs      : 100
Testable  : 38 instructions
================================================================
[42]  ADDI    addi x19, x12, 893              match
[43]  XOR     xor x7, x22, x14               match
...
================================================================
runs=100  match=100  divergence=0  error=0
```

> **Note:** Load instructions are excluded by default (sandbox memory
> is uninitialised on Proteus vs zero on Unicorn → false divergences).

---

### Step 5 — Level 3a: CT-SEQ contract checking

Requires Docker running.

```bash
# Without loads (default)
python3 contract_checker.py --seed 42 --runs 50

# With loads (detects data-dependent memory leaks)
python3 contract_checker.py --seed 42 --runs 50 --include-loads
```

Expected output with `--include-loads`:
```
[42]  LB      lb x21, 0(x25)                 <<< VIOLATION
────────────────────────────────────────────────────────────────
VIOLATION  seed=42  LB  lb x21, 0(x25)
  event[5]: A=(mem, 0x800012f4)  B=(mem, 0x80001278)
```

---

### Step 6 — Level 3b: CT-COND Spectre-v1 detection

Requires Docker running.

```bash
# One gadget with full trace output
python3 cond_checker.py --seed 42

# 30 random gadgets
python3 cond_checker.py --runs 30 --seed 42

# Narrower speculative window
python3 cond_checker.py --runs 30 --seed 42 --window 3
```

Expected output:
```
CT-COND checker  (window=8)
Base seed : 42   runs: 30
====================================================================
[42]  VIOLATION  and x7, x17, x11 | bltu x7, x11, .skip | lw x12, 0(x7)
[43]  CT-safe    sub x9, x14, x22 | bne  x9, x0, .skip  | lb x6, 0(x9)
...
====================================================================
runs=30  CT-safe=22  violation=8  error=0
```

---

## Historical PoCs

Three early proof-of-concept scripts document the evolution of the project
(Phase 2 above). They are kept as witnesses but are **not part of the
production pipeline**.

| Script | What it shows |
|---|---|
| `riscv_unicorn_poc.py` | First Unicorn execution of a RV32 instruction |
| `riscv_unicorn_poc2.py` | Named operand lookup + register diff after execution |
| `riscv_unicorn_loadstores_tb.py` | Memory sandbox — loads and stores work correctly |

```bash
# All three accept --seed and --json
python3 riscv_unicorn_poc.py --seed 42
python3 riscv_unicorn_poc2.py --seed 42
python3 riscv_unicorn_loadstores_tb.py --seed 42
```

> These PoCs use `riscv64-elf-as` locally (not Docker) and run in
> `UC_MODE_RISCV64`. They predate the switch to RV32 and Proteus.

---

## Reproducing results

Every script that uses randomness accepts `--seed`. The same seed always
produces the same program and initial register state on any machine.

```bash
# Reproduce the CT-SEQ violation from the paper
python3 contract_checker.py --seed 42 --include-loads

# Reproduce the Spectre-v1 results (8/30 gadgets)
python3 cond_checker.py --runs 30 --seed 0
```

---

## Architecture summary

```
riscv_isa_rv32.json
       │
       ▼
format_instruction()          ← random RV32IM instruction
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
  Unicorn (UC_MODE_RISCV32)           Docker (Proteus RTL)
  Pure ISA model                      Cycle-accurate OoO pipeline
  reg_read() → register state         REGDUMP + MEMTRACE → stdout
       │                                      │
       └──────────────┬───────────────────────┘
                      ▼
              Compare x5–x29
              ──────────────
              match      → Proteus is ISA-correct
              divergence → functional bug in Proteus
              CT violation → information leak detected
```

---

## Troubleshooting

**Docker daemon not running**
```bash
open -a Docker          # macOS: start Docker Desktop
docker info | head -3   # verify it's up
```

**`riscv64-elf-as` not found** (needed for `test_instructions.py` and PoCs)
```bash
brew tap riscv-software-src/riscv
brew install riscv-tools
```

**Proteus simulation timeout (>30s)**
The CharDev EOT halt (`sb x31, 0(x30)` writing byte `0x04` to `0x10000000`)
must be present in the program. If missing, Proteus runs until `MAX_CYCLES`
(~77 seconds). All scripts in this repo include the halt correctly.

**No REGDUMP in output**
Rebuild the Docker image — the simulator must include the REGDUMP block
added to `main.cpp`:
```bash
cd proteus
docker build -t proteus .
```

**False divergences in diff_tester.py**
Load instructions read from sandbox memory which is uninitialised on Proteus
(`0xcafebabe`) but zero on Unicorn. Loads are excluded by default for this
reason. Use `--include-loads` only in `contract_checker.py` where the input
state is set explicitly via `reg_write`.

---

## References

- Guarnieri et al., *Hardware-Software Contracts for Secure Speculation*, IEEE S&P 2021
- García Arribas, *Fuzzing RISC-V Processors for Speculative Leaks*, TFG UPM 2023
- Oleksenko et al., *Revizor: Testing Black-Box CPUs Against Speculation Contracts*, ASPLOS 2022
