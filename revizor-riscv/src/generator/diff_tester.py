"""
Differential tester: runs the same RV32IM program on Proteus and Unicorn,
then compares the resulting register state (x5..x29) to detect divergences.

How it works:
  - A shared core (prologue + deterministic init + test instruction) is compiled
    once and used identically by both executors.
  - Proteus variant: core + register dump to DUMP_ADDR + EOT halt via CharDev.
  - Unicorn variant: core only — registers are read directly after emulation.
  - Both binaries are compiled inside a single Docker call for consistency.

Current scope:
  - Excludes control-flow instructions (require labels, separate phase).
  - Excludes LOADs: sandbox memory is uninitialised on Proteus (0xcafebabe)
    vs zero-filled on Unicorn, which would produce false divergences.

Usage:
    python3 diff_tester.py
    python3 diff_tester.py --runs 100 --seed 42
    python3 diff_tester.py --isa riscv_isa_rv32.json --runs 10
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

from unicorn import Uc, UC_ARCH_RISCV, UC_MODE_RISCV32, UcError
from unicorn.riscv_const import *

RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")

DOCKER_IMAGE     = "proteus"
RAM_BASE         = 0x80000000   # code + data base address
SANDBOX_ADDR     = 0x80001000   # memory sandbox base, held in x30
DUMP_ADDR        = 0x80002000   # Proteus dumps x5..x29 here after execution
RAM_SIZE_UNICORN = 0x00100000   # 1 MiB flat region, mirrors Proteus RAM

RESERVED_REGS = {"x30", "x31"}      # x30 = sandbox base, x31 = dump pointer
TRACKED       = list(range(5, 30))  # registers compared: x5..x29

UC_REG = {i: globals()[f"UC_RISCV_REG_X{i}"] for i in range(32)}

ERROR_OUTPUT_LIMIT = 800  # max chars of Docker output shown on error


# ── Instruction generation ───────────────────────────────────────────────────

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
    """Render one JSON entry as an assembly line."""
    name = instr["name"].lower()
    cat  = instr["category"]
    ops  = instr["operands"]

    if "STORE" in cat:
        rs2    = pick_register(get_op(ops, "rs2"))
        offset = pick_immediate(get_op(ops, "offset"))
        # Align offset to access size to avoid alignment traps on Proteus.
        if   name == "sw": offset = (offset // 4) * 4
        elif name == "sh": offset = (offset // 2) * 2
        # sb: byte-aligned by definition
        return f"{name} {rs2}, {offset}(x30)"

    if "UPPER-IMM" in cat:
        rd  = pick_register(get_op(ops, "rd"), exclude=RESERVED_REGS)
        imm = pick_immediate(get_op(ops, "imm20"))
        return f"{name} {rd}, {imm}"

    # Default: positional REG/IMM operands.
    parts = []
    for op in ops:
        if op["type_"] == "REG":
            excl = RESERVED_REGS if op.get("dest") else None
            parts.append(pick_register(op, exclude=excl))
        elif op["type_"] == "IMM":
            parts.append(str(pick_immediate(op)))
    return f"{name} {', '.join(parts)}"


# ── Assembly construction ────────────────────────────────────────────────────

def build_core(instr_line, init_vals):
    """Prologue + deterministic register init + test instruction. Shared by both executors."""
    hi   = (SANDBOX_ADDR >> 12) & 0xFFFFF
    lo   = SANDBOX_ADDR & 0xFFF
    init = "\n".join(f"    li x{i}, 0x{init_vals[i]:08x}" for i in TRACKED)
    return (
        ".section .text\n"
        ".globl _start\n"
        "_start:\n"
        f"    lui   x30, {hi}\n"
        f"    addi  x30, x30, {lo}\n"
        f"\n{init}\n\n"
        f"    {instr_line}\n"
    )


def build_proteus(core):
    """Append register dump to DUMP_ADDR and EOT halt to the core program."""
    stores = "\n".join(f"    sw   x{r}, {idx * 4}(x31)" for idx, r in enumerate(TRACKED))
    return (
        core
        + "\n    # dump x5..x29 to DUMP_ADDR via x31 (reserved dump pointer)\n"
        + "    lui  x31, 0x80002\n"
        + stores + "\n"
        + "\n    # halt via CharDev EOT byte\n"
        + "    lui  x30, 0x10000\n"
        + "    li   x31, 4\n"
        + "    sb   x31, 0(x30)\n"
        + ".Lhalt: j .Lhalt\n"
    )


# ── Compile + run ────────────────────────────────────────────────────────────

def build_and_run(proteus_src, unicorn_src):
    """
    Compile both programs inside Docker and run the Proteus binary.
    Returns (proteus_output, unicorn_binary_bytes, proteus_binary_bytes).
    """
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "p.S").write_text(proteus_src, encoding="utf-8")
        Path(tmp, "u.S").write_text(unicorn_src, encoding="utf-8")

        gcc = (
            "riscv32-unknown-elf-gcc -march=rv32im_zicsr -mabi=ilp32 "
            "-ffreestanding -nostdlib -Wl,--no-relax -Wl,-Ttext=0x80000000"
        )
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


# ── State extraction ─────────────────────────────────────────────────────────

def parse_regdump(output):
    """Parse {reg_index: value} from the [REGDUMP] block in Proteus output."""
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
    """Run the core binary in Unicorn RV32. Returns {reg_index: value} or None."""
    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV32)
        mu.mem_map(RAM_BASE, RAM_SIZE_UNICORN)
        mu.mem_write(RAM_BASE, code)
        mu.emu_start(RAM_BASE, RAM_BASE + len(code), timeout=1_000_000, count=0)
        return {i: mu.reg_read(UC_REG[i]) & 0xFFFFFFFF for i in TRACKED}
    except UcError as e:
        print(f"[unicorn] error: {e}")
        return None


# ── Single differential test ─────────────────────────────────────────────────

def run_one(instr, seed):
    """
    Run one differential test. Returns (status, detail).
    status in {'match', 'divergence', 'error'}.
    """
    random.seed(seed)
    instr_line = format_instruction(instr)
    init_vals  = {i: random.randrange(0, 1 << 32) for i in TRACKED}

    core = build_core(instr_line, init_vals)
    pout, ubin, pbin = build_and_run(build_proteus(core), core)

    if "[TIMEOUT]" in pout or ubin is None:
        return "error", (instr_line, "compile/sim failed", pout)

    pstate = parse_regdump(pout)
    if pstate is None:
        return "error", (instr_line, "no REGDUMP in Proteus output", pout)

    # Sanity: the shared core must encode identically in both binaries.
    if pbin is not None and not pbin.startswith(ubin):
        print("   [warn] core prefix differs between variants (encoding drift)")

    ustate = run_on_unicorn(ubin)
    if ustate is None:
        return "error", (instr_line, "unicorn failed", pout)

    diffs = [
        (i, pstate.get(i), ustate.get(i))
        for i in TRACKED
        if pstate.get(i) != ustate.get(i)
    ]
    if diffs:
        return "divergence", (instr_line, diffs, pstate, ustate)
    return "match", (instr_line, pstate, ustate)


# ── Output helpers ───────────────────────────────────────────────────────────

def print_table(pstate, ustate):
    print("\n" + "-" * 64)
    print(f"{'reg':<5}{'Proteus':<14}{'Unicorn':<14}status")
    print("-" * 64)
    for i in TRACKED:
        p, u = pstate.get(i, 0), ustate.get(i, 0)
        print(f"x{i:<4}0x{p:08x}    0x{u:08x}    {'ok' if p == u else '<<< MISMATCH'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Differential tester: Proteus vs Unicorn (RV32IM)")
    ap.add_argument("--seed", type=int, default=None,                help="Base seed (random if omitted)")
    ap.add_argument("--runs", type=int, default=1,                   help="Number of test cases")
    ap.add_argument("--isa",  default="riscv_isa_rv32.json",         help="ISA spec file")
    args = ap.parse_args()

    base_seed = args.seed if args.seed is not None else random.randrange(1 << 30)

    with open(args.isa, encoding="utf-8") as f:
        instructions = json.load(f)

    # Exclude control-flow (need labels) and LOADs (memory not synchronised).
    testable = [
        i for i in instructions
        if not i.get("is_control_flow", False)
        and "LOAD" not in i["category"]
    ]

    print(f"Base seed : {base_seed}")
    print(f"Runs      : {args.runs}")
    print(f"Testable  : {len(testable)} instructions")
    print("=" * 64)

    n_match = n_diff = n_err = 0
    divergences  = []
    last_result  = None

    for k in range(args.runs):
        seed   = base_seed + k
        random.seed(seed)
        instr  = random.choice(testable)
        status, detail = run_one(instr, seed)
        last_result = (status, instr, detail)
        line = detail[0]

        if status == "match":
            n_match += 1
            print(f"[{seed}] {instr['name']:7s}  {line:32s}  match")
        elif status == "divergence":
            n_diff += 1
            divergences.append((seed, instr, detail))
            print(f"[{seed}] {instr['name']:7s}  {line:32s}  <<< DIVERGENCE")
        else:
            n_err += 1
            print(f"[{seed}] {instr['name']:7s}  {line:32s}  ERROR ({detail[1]})")

    # For a single run, show the full register table.
    if args.runs == 1:
        status, instr, detail = last_result
        if status in ("match", "divergence"):
            pstate = detail[2] if status == "divergence" else detail[1]
            ustate = detail[3] if status == "divergence" else detail[2]
            print_table(pstate, ustate)
        else:
            print("\n[error detail]")
            print(detail[2][:ERROR_OUTPUT_LIMIT])

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