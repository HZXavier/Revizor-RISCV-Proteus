"""
Differential tester: runs the same RV32IM program on Proteus and Unicorn,
then compares the resulting register state (x5..x29) to detect divergences.

Why this is valid:
  - The prologue + initial-state block + test instruction are written ONCE and
    compiled identically for both executors, so the test instruction runs from a
    byte-identical state on each side.
  - Proteus can only be observed through memory, so its variant appends a dump
    block (store x5..x29 to DUMP_ADDR) and halts via the CharDev EOT.
  - Unicorn exposes registers directly, so its variant stops right after the
    test instruction and reads them with reg_read -- no dump, no halt loop
    (the halt loop would hang Unicorn since it has no CharDev).

Scope of this version:
  - Excludes control-flow (needs labels / a separate phase).
  - Excludes LOAD: a load reads sandbox memory, which is 0xcafebabe on Proteus
    (uninitialised fill) vs 0x0 on Unicorn -> would be a false divergence.
    Loads come in phase 2b once memory is synchronised on both sides.
"""
import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

from unicorn import Uc, UC_ARCH_RISCV, UC_MODE_RISCV32, UcError
from unicorn.riscv_const import *  # UC_RISCV_REG_X*

RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")

DOCKER_IMAGE     = "proteus"
RAM_BASE         = 0x80000000      # code base (both) / Proteus RAM base
SANDBOX_ADDR     = 0x80001000      # base for loads/stores, held in x30
DUMP_ADDR        = 0x80002000      # Proteus reads registers back from here
RAM_SIZE_UNICORN = 0x00100000      # 1 MiB flat region, mirrors Proteus RAM

RESERVED_REGS = {"x30", "x31"}     # sandbox base + aux, never used as dest
TRACKED = list(range(5, 30))       # registers compared: x5..x29

# Map register index -> Unicorn register id, once.
UC_REG = {i: globals()[f"UC_RISCV_REG_X{i}"] for i in range(32)}


# ---------------------------------------------------------------------------
# Instruction generation
# ---------------------------------------------------------------------------
def pick_register(operand, exclude=None):
    candidates = [v for v in operand["values"] if v not in (exclude or set())]
    if not candidates:
        raise ValueError(f"No registers left after excluding {exclude}")
    return random.choice(candidates)


def pick_immediate(operand):
    m = RANGE_RE.match(operand["values"][0])
    return random.randint(int(m.group(1)), int(m.group(2)))


def get_op(operands, name):
    for op in operands:
        if op["name"] == name:
            return op
    raise KeyError(name)

def format_instruction(instr):
    name = instr["name"].lower()
    cat  = instr["category"]
    ops  = instr["operands"]

    if "STORE" in cat:
        rs2    = pick_register(get_op(ops, "rs2"))
        offset = pick_immediate(get_op(ops, "offset"))
        # align offset to access size to avoid alignment trap on Proteus
        if name == "sw":
            offset = (offset // 4) * 4
        elif name == "sh":
            offset = (offset // 2) * 2
        # sb: byte access, always aligned
        return f"{name} {rs2}, {offset}(x30)"

    if "UPPER-IMM" in cat:
        rd  = pick_register(get_op(ops, "rd"), exclude=RESERVED_REGS)
        imm = pick_immediate(get_op(ops, "imm20"))
        return f"{name} {rd}, {imm}"

    parts = []
    for op in ops:
        if op["type_"] == "REG":
            excl = RESERVED_REGS if op.get("dest") else None
            parts.append(pick_register(op, exclude=excl))
        elif op["type_"] == "IMM":
            parts.append(str(pick_immediate(op)))
    return f"{name} {', '.join(parts)}"

# ---------------------------------------------------------------------------
# Assembly construction
# ---------------------------------------------------------------------------
def build_core(instr_line, init_vals):
    """Prologue + deterministic init + test instruction. Shared by both executors."""
    hi = (SANDBOX_ADDR >> 12) & 0xFFFFF
    lo = SANDBOX_ADDR & 0xFFF
    init = "\n".join(f"    li x{i}, 0x{init_vals[i]:08x}" for i in TRACKED)
    return f""".section .text
.globl _start
_start:
    # sandbox base -> x30 (same in both executors)
    lui   x30, {hi}
    addi  x30, x30, {lo}

    # deterministic initial state for x5..x29
{init}

    # === TEST INSTRUCTION ===
    {instr_line}
    # ========================
"""

def build_proteus(core):
    """Proteus variant: core + register dump + EOT halt."""
    stores = "\n".join(f"    sw   x{r}, {idx * 4}(x31)" for idx, r in enumerate(TRACKED))
    return core + f"""
    # dump x5..x29 to DUMP_ADDR — use x31 as pointer (reserved, never clobbered)
    lui  x31, 0x80002
{stores}

    # halt: write EOT to CharDev — use x30/x31 (both reserved)
    lui  x30, 0x10000
    li   x31, 4
    sb   x31, 0(x30)
1:  j 1b
"""


def build_unicorn(core):
    """Unicorn variant: core only. Registers are read directly after the test."""
    return core


# ---------------------------------------------------------------------------
# Compile both + run Proteus, in a single Docker invocation
# ---------------------------------------------------------------------------

def build_and_run(proteus_src, unicorn_src):
    """Returns (proteus_output, unicorn_bin_bytes, proteus_bin_bytes)."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "p.S").write_text(proteus_src)
        Path(tmp, "u.S").write_text(unicorn_src)

        gcc = ("riscv32-unknown-elf-gcc -march=rv32im_zicsr -mabi=ilp32 "
               "-ffreestanding -nostdlib -Wl,--no-relax -Wl,-Ttext=0x80000000")
        script = " && ".join([
            f"{gcc} -o /work/p.elf /work/p.S",
            "riscv32-unknown-elf-objcopy -O binary /work/p.elf /work/p.bin",
            f"{gcc} -o /work/u.elf /work/u.S",
            "riscv32-unknown-elf-objcopy -O binary /work/u.elf /work/u.bin",
            "/proteus/sim/build/sim /work/p.bin",
        ])
        try:
            r = subprocess.run(
                ["docker", "run", "--rm", "-v", f"{tmp}:/work",
                 DOCKER_IMAGE, "bash", "-c", script],
                capture_output=True, text=True, timeout=30,
            )
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            out = "[TIMEOUT] Proteus simulation exceeded 30s"

        ubin = Path(tmp, "u.bin").read_bytes() if Path(tmp, "u.bin").exists() else None
        pbin = Path(tmp, "p.bin").read_bytes() if Path(tmp, "p.bin").exists() else None
        return out, ubin, pbin

# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------
def parse_regdump(output):
    """Extract {reg_index: value} from the [REGDUMP] block, or None."""
    if "[REGDUMP]" not in output or "[/REGDUMP]" not in output:
        return None
    body = output[output.index("[REGDUMP]") + len("[REGDUMP]"):output.index("[/REGDUMP]")]
    state = {}
    for line in body.splitlines():
        line = line.strip()
        if "=" in line:
            reg, val = line.split(" = ")
            state[int(reg.strip()[1:])] = int(val, 16) & 0xFFFFFFFF
    return state


def run_on_unicorn(code):
    """Run the core binary in Unicorn RV32; return {reg_index: value} or None."""
    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV32)
        mu.mem_map(RAM_BASE, RAM_SIZE_UNICORN)   # single flat region, mirrors Proteus
        mu.mem_write(RAM_BASE, code)
        # stop right after the last (test) instruction; timeout is a safety net
        mu.emu_start(RAM_BASE, RAM_BASE + len(code), timeout=1_000_000, count=0)
        return {i: mu.reg_read(UC_REG[i]) & 0xFFFFFFFF for i in TRACKED}
    except UcError as e:
        print(f"[unicorn] error: {e}")
        return None


# ---------------------------------------------------------------------------
# One differential test
# ---------------------------------------------------------------------------
def run_one(instr, seed, warn_prefix=True):
    """Returns (status, detail). status in {'match','divergence','error'}."""
    random.seed(seed)
    instr_line = format_instruction(instr)
    init_vals  = {i: random.randrange(0, 1 << 32) for i in TRACKED}

    core = build_core(instr_line, init_vals)
    pout, ubin, pbin = build_and_run(build_proteus(core), build_unicorn(core))
    if "[TIMEOUT]" in pout or ubin is None:
        return "error", (instr_line, "compile/sim failed", pout)

    pstate = parse_regdump(pout)
    if pstate is None or ubin is None:
        return "error", (instr_line, "compile/sim failed", pout)

    # sanity: the shared core must encode identically in both binaries
    if warn_prefix and pbin is not None and not pbin.startswith(ubin):
        print("   [warn] core prefix differs between variants (encoding drift)")

    ustate = run_on_unicorn(ubin)
    if ustate is None:
        return "error", (instr_line, "unicorn failed", pout)

    diffs = [(i, pstate.get(i), ustate.get(i)) for i in TRACKED if pstate.get(i) != ustate.get(i)]
    if diffs:
        return "divergence", (instr_line, diffs, pstate, ustate)
    return "match", (instr_line, pstate, ustate)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_table(pstate, ustate):
    print("\n" + "-" * 64)
    print(f"{'reg':<5}{'Proteus':<14}{'Unicorn':<14}status")
    print("-" * 64)
    for i in TRACKED:
        p, u = pstate.get(i, 0), ustate.get(i, 0)
        mark = "ok" if p == u else "<<< MISMATCH"
        print(f"x{i:<4}0x{p:08x}    0x{u:08x}    {mark}")


def main():
    ap = argparse.ArgumentParser(description="Differential tester: Proteus vs Unicorn (RV32IM)")
    ap.add_argument("--seed", type=int, default=None, help="base seed (random if omitted)")
    ap.add_argument("--runs", type=int, default=1, help="number of test cases")
    ap.add_argument("--isa", default="riscv_isa_rv32.json", help="ISA spec file")
    args = ap.parse_args()

    base_seed = args.seed if args.seed is not None else random.randrange(1 << 30)

    with open(args.isa) as f:
        instructions = json.load(f)

    testable = [i for i in instructions
                if not i.get("is_control_flow", False)
                and "LOAD" not in i["category"]]   # loads need memory sync (phase 2b)

    print(f"Base seed: {base_seed}   runs: {args.runs}   testable instrs: {len(testable)}")
    print("=" * 64)

    n_match = n_diff = n_err = 0
    divergences = []
    last = None  # (status, instr, detail) of the most recent run, for the single-run table

    for k in range(args.runs):
        seed = base_seed + k
        random.seed(seed)
        instr = random.choice(testable)
        status, detail = run_one(instr, seed)
        last = (status, instr, detail)
        line = detail[0]

        if status == "match":
            n_match += 1
            print(f"[{seed}] {instr['name']:7s} {line:32s} match")
        elif status == "divergence":
            n_diff += 1
            divergences.append((seed, instr, detail))
            print(f"[{seed}] {instr['name']:7s} {line:32s} <<< DIVERGENCE")
        else:
            n_err += 1
            print(f"[{seed}] {instr['name']:7s} {line:32s} ERROR ({detail[1]})")

    # Single-run: show the full register table (reusing the computed result)
    if args.runs == 1 and last is not None:
        status, instr, detail = last
        if status == "match":
            _, pstate, ustate = detail
            print_table(pstate, ustate)
        elif status == "divergence":
            _, diffs, pstate, ustate = detail
            print_table(pstate, ustate)
        elif status == "error":
            print("\n[error detail]")
            print(detail[2][:600])

    print("\n" + "=" * 64)
    print(f"runs={args.runs}  match={n_match}  divergence={n_diff}  error={n_err}")

    for seed, instr, detail in divergences:
        line, diffs = detail[0], detail[1]
        print("\n" + "-" * 64)
        print(f"DIVERGENCE  seed={seed}  {instr['name']}  {line}")
        for i, p, u in diffs:
            ps = f"0x{p:08x}" if p is not None else "----"
            us = f"0x{u:08x}" if u is not None else "----"
            print(f"  x{i:<2}: proteus={ps}  unicorn={us}")


if __name__ == "__main__":
    main()