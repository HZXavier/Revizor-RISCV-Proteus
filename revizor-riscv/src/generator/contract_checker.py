"""
CT-SEQ contract checker: Constant-Time contract with Sequential execution clause.

Based on Garcia 2023 (TFG section 3.3) and Guarnieri et al. 2021.
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

from unicorn import Uc, UC_ARCH_RISCV, UC_MODE_RISCV32, UcError
from unicorn import UC_HOOK_CODE, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE
from unicorn.riscv_const import *

RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")

DOCKER_IMAGE = "proteus"
RAM_BASE     = 0x80000000
SANDBOX_ADDR = 0x80001000   # sandbox base, always held in x30
RAM_SIZE     = 0x00100000   # 1 MiB

RESERVED_REGS = {"x30", "x31"}
TRACKED       = list(range(5, 30))
UC_REG        = {i: globals()[f"UC_RISCV_REG_X{i}"] for i in range(32)}


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

    if "LOAD" in cat:
        rd  = pick_register(get_op(ops, "rd"),  exclude=RESERVED_REGS)
        rs1 = pick_register(get_op(ops, "rs1"), exclude=RESERVED_REGS)
        # offset=0: address = rs1 value, which varies between inputs -> CT violation possible
        return f"{name} {rd}, 0({rs1})"

    if "STORE" in cat:
        rs2    = pick_register(get_op(ops, "rs2"))
        offset = pick_immediate(get_op(ops, "offset"))
        if   name == "sw": offset = (offset // 4) * 4
        elif name == "sh": offset = (offset // 2) * 2
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


# ── Assembly + compilation ───────────────────────────────────────────────────

def build_core_ct(instr_line):
    """Minimal program for CT checking. Register state is set via reg_write, not inline."""
    return (
        ".section .text\n"
        ".globl _start\n"
        "_start:\n"
        f"    {instr_line}\n"
    )


def compile_core(asm_source):
    """Compile to raw binary via Docker. Returns bytes or None on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "t.S").write_text(asm_source, encoding="utf-8")

        r = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{tmp}:/work", DOCKER_IMAGE,
             "riscv32-unknown-elf-gcc",
             "-march=rv32im_zicsr", "-mabi=ilp32",
             "-ffreestanding", "-nostdlib",
             "-Wl,--no-relax", "-Wl,-Ttext=0x80000000",
             "-o", "/work/t.elf", "/work/t.S"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return None

        r = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{tmp}:/work", DOCKER_IMAGE,
             "riscv32-unknown-elf-objcopy", "-O", "binary",
             "/work/t.elf", "/work/t.bin"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return None

        return Path(tmp, "t.bin").read_bytes()


# ── CT-SEQ trace collection ──────────────────────────────────────────────────

def collect_ct_trace(code, init_vals):
    """
    Run code in Unicorn with CT hooks and return the observation trace.
    Traces: list of ("pc", addr) and ("mem", addr) events.
    """
    trace = []

    def hook_code(uc, address, size, user_data):
        trace.append(("pc", address))

    def hook_mem(uc, access, address, size, value, user_data):
        trace.append(("mem", address))

    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV32)
        mu.mem_map(RAM_BASE, RAM_SIZE)
        mu.mem_write(RAM_BASE, code)
        mu.mem_write(SANDBOX_ADDR & ~0xFFF,
                     b"\xab\xcd\xef\x01" * (RAM_SIZE // 4 - 0x1000 // 4))

        # Set register state directly — no prologue instructions.
        for i in TRACKED:
            mu.reg_write(UC_REG[i], init_vals[i])
        mu.reg_write(UC_REG[30], SANDBOX_ADDR)

        mu.hook_add(UC_HOOK_CODE, hook_code)
        mu.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, hook_mem)

        mu.emu_start(RAM_BASE, RAM_BASE + len(code), timeout=0, count=0)

    except UcError:
        pass

    return trace


# ── Two-input CT-SEQ check ───────────────────────────────────────────────────

def make_input_pair(seed):
    """Generate two inputs where tracked registers point into the sandbox at different offsets."""
    rng = random.Random(seed)
    state_a = {i: SANDBOX_ADDR + (rng.randrange(0, 0x400) & ~3) for i in TRACKED}
    state_b = {i: SANDBOX_ADDR + (rng.randrange(0, 0x400) & ~3) for i in TRACKED}
    return state_a, state_b


def check_ct(code, seed):
    """
    Run the CT-SEQ check for one instruction with two inputs.
    Returns ('safe', ta, tb, []) or ('violation', ta, tb, diffs).
    """
    state_a, state_b = make_input_pair(seed)
    ta = collect_ct_trace(code, state_a)
    tb = collect_ct_trace(code, state_b)

    min_len = min(len(ta), len(tb))
    diffs   = [(i, ta[i], tb[i]) for i in range(min_len) if ta[i] != tb[i]]

    if diffs or len(ta) != len(tb):
        return "violation", ta, tb, diffs
    return "safe", ta, tb, []


# ── Output ───────────────────────────────────────────────────────────────────

def print_trace(trace, label, max_events=20):
    print(f"\n--- CT trace for input {label} ---")
    for ev in trace[:max_events]:
        print(f"  {ev[0]:3s}  0x{ev[1]:08x}")
    if len(trace) > max_events:
        print(f"  ... ({len(trace)} events total)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="CT-SEQ contract checker (Unicorn)")
    ap.add_argument("--seed",          type=int, default=None)
    ap.add_argument("--runs",          type=int, default=1)
    ap.add_argument("--isa",           default="riscv_isa_rv32.json")
    ap.add_argument("--include-loads", action="store_true",
                    help="Include load instructions (excluded by default)")
    args = ap.parse_args()

    base_seed = args.seed if args.seed is not None else random.randrange(1 << 30)

    with open(args.isa, encoding="utf-8") as f:
        instructions = json.load(f)

    # Exclude control-flow (branches need labels) and loads unless requested.
    testable = [i for i in instructions if not i.get("is_control_flow", False)]
    if not args.include_loads:
        testable = [i for i in testable if "LOAD" not in i["category"]]

    print(f"CT-SEQ contract checker")
    print(f"Base seed : {base_seed}   runs: {args.runs}   testable: {len(testable)}")
    print("=" * 68)

    n_safe = n_viol = n_err = 0
    violations  = []
    last_result = None

    for k in range(args.runs):
        seed = base_seed + k
        random.seed(seed)
        instr = random.choice(testable)

        try:
            instr_line = format_instruction(instr)
        except Exception as e:
            print(f"[{seed}] {instr['name']:7s} ERROR (format: {e})")
            n_err += 1
            continue

        code = compile_core(build_core_ct(instr_line))
        if code is None:
            print(f"[{seed}] {instr['name']:7s} {instr_line:35s} ERROR (compile)")
            n_err += 1
            continue

        status, ta, tb, diffs = check_ct(code, seed)
        last_result = (status, instr_line, ta, tb, diffs)

        if status == "safe":
            n_safe += 1
            print(f"[{seed}] {instr['name']:7s} {instr_line:35s} CT-safe")
        else:
            n_viol += 1
            violations.append((seed, instr, instr_line, ta, tb, diffs))
            print(f"[{seed}] {instr['name']:7s} {instr_line:35s} <<< VIOLATION")

    # For a single run, show both traces.
    if args.runs == 1 and last_result is not None:
        status, instr_line, ta, tb, diffs = last_result
        print_trace(ta, "A")
        print_trace(tb, "B")

    print("\n" + "=" * 68)
    print(f"runs={args.runs}  CT-safe={n_safe}  violation={n_viol}  error={n_err}")

    for seed, instr, line, ta, tb, diffs in violations:
        print(f"\n{'─' * 68}")
        print(f"VIOLATION  seed={seed}  {instr['name']}  {line}")
        print("  First diverging events:")
        for i, ea, eb in diffs[:5]:
            print(f"    event[{i}]: A=({ea[0]}, 0x{ea[1]:08x})  B=({eb[0]}, 0x{eb[1]:08x})")
        if len(diffs) > 5:
            print(f"    ... ({len(diffs)} diverging events total)")


if __name__ == "__main__":
    main()