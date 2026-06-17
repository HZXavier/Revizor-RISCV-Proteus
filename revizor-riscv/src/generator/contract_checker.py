"""
Contract checker: CT-SEQ leakage contract (Constant-Time, Sequential clause).

Based on the contract framework from Eric Garcia's TFG (section 3.3) and
the hardware-software contracts for secure speculation (Guarnieri et al. 2021).

The CT-SEQ contract states:
  Two executions of the same program with different inputs must produce the
  same observation trace (PC sequence + memory access addresses).
  A difference in traces = a potential information leak.

The observation clause CT exposes:
  - PC value after each instruction
  - Address of every memory load and store

The execution clause SEQ means:
  - No speculation: traces collected during normal sequential execution only.

Usage:
  python3 contract_checker.py               # one random test
  python3 contract_checker.py --runs 50     # fuzz 50 cases
  python3 contract_checker.py --seed 42     # reproducible run
  python3 contract_checker.py --include-loads  # also test load instructions
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

DOCKER_IMAGE  = "proteus"
RAM_BASE      = 0x80000000
SANDBOX_ADDR  = 0x80001000      # x30 = sandbox base for loads/stores
RAM_SIZE      = 0x00100000      # 1 MiB

RESERVED_REGS = {"x30", "x31"}
TRACKED       = list(range(5, 30))

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

    if "LOAD" in cat:
        rd  = pick_register(get_op(ops, "rd"), exclude=RESERVED_REGS)
        rs1 = pick_register(get_op(ops, "rs1"), exclude=RESERVED_REGS)
        # offset=0, address varies with rs1 (data-dependent → CT violation possible)
        return f"{name} {rd}, 0({rs1})"

    if "STORE" in cat:
        rs2    = pick_register(get_op(ops, "rs2"))
        offset = pick_immediate(get_op(ops, "offset"))
        if name == "sw":
            offset = (offset // 4) * 4
        elif name == "sh":
            offset = (offset // 2) * 2
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
# Assembly + compilation (shared core for both inputs)
# ---------------------------------------------------------------------------
def build_core_ct(instr_line):
    """Program for CT checking: no register init (state set via reg_write)."""
    hi = (SANDBOX_ADDR >> 12) & 0xFFFFF
    lo = SANDBOX_ADDR & 0xFFF
    return f""".section .text
.globl _start
_start:
    # x30 set via reg_write in Unicorn, no li block needed
    {instr_line}
"""

def compile_core(asm_source):
    """Compile to raw binary via Docker. Returns bytes or None."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "t.S").write_text(asm_source)
        gcc = ["riscv32-unknown-elf-gcc", "-march=rv32im_zicsr", "-mabi=ilp32",
               "-ffreestanding", "-nostdlib", "-Wl,--no-relax",
               "-Wl,-Ttext=0x80000000", "-o", "/work/t.elf", "/work/t.S"]
        r = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{tmp}:/work", DOCKER_IMAGE] + gcc,
            capture_output=True, text=True)
        if r.returncode != 0:
            return None
        r = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{tmp}:/work", DOCKER_IMAGE,
             "riscv32-unknown-elf-objcopy", "-O", "binary",
             "/work/t.elf", "/work/t.bin"],
            capture_output=True, text=True)
        if r.returncode != 0:
            return None
        return Path(tmp, "t.bin").read_bytes()


# ---------------------------------------------------------------------------
# CT trace collection via Unicorn hooks
# ---------------------------------------------------------------------------
def collect_ct_trace(code, init_vals):
    trace = []

    def hook_code(uc, address, size, user_data):
        trace.append(("pc", address))

    def hook_mem(uc, access, address, size, value, user_data):
        trace.append(("mem", address))

    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV32)
        mu.mem_map(RAM_BASE, RAM_SIZE)
        mu.mem_write(RAM_BASE, code)

        # fill sandbox with neutral pattern
        mu.mem_write(SANDBOX_ADDR & ~0xFFF,
                     b"\xab\xcd\xef\x01" * (RAM_SIZE // 4 - 0x1000 // 4))

        # write ALL tracked registers directly — no prologue overwrites them
        for i in TRACKED:
            mu.reg_write(UC_REG[i], init_vals[i])
        mu.reg_write(UC_REG[30], SANDBOX_ADDR)  # sandbox base always fixed

        mu.hook_add(UC_HOOK_CODE, hook_code)
        mu.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, hook_mem)

        mu.emu_start(RAM_BASE, RAM_BASE + len(code), timeout=0, count=0)

    except UcError:
        pass

    return trace

# ---------------------------------------------------------------------------
# Two-input CT test
# ---------------------------------------------------------------------------
def make_input_pair(instr, base_seed):
    rng = random.Random(base_seed)
    state_a = {}
    state_b = {}
    for i in TRACKED:
        off_a = (rng.randrange(0, 0x400)) & ~3
        off_b = (rng.randrange(0, 0x400)) & ~3
        state_a[i] = SANDBOX_ADDR + off_a
        state_b[i] = SANDBOX_ADDR + off_b
    return state_a, state_b

def check_ct(instr_line, code, seed):
    """
    Run the CT-SEQ check for one instruction.
    Returns ('safe', ...) or ('violation', trace_a, trace_b, diff).
    """
    state_a, state_b = make_input_pair(None, seed)

    trace_a = collect_ct_trace(code, state_a)
    trace_b = collect_ct_trace(code, state_b)

    # Normalise trace length (a fault may cut one short)
    min_len = min(len(trace_a), len(trace_b))
    ta = trace_a[:min_len]
    tb = trace_b[:min_len]

    diffs = [(i, ta[i], tb[i]) for i in range(min_len) if ta[i] != tb[i]]

    if diffs or len(trace_a) != len(trace_b):
        return "violation", trace_a, trace_b, diffs
    return "safe", trace_a, trace_b, []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CT-SEQ contract checker (Unicorn)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--isa", default="riscv_isa_rv32.json")
    ap.add_argument("--include-loads", action="store_true",
                    help="Also test load instructions (included by default in CT mode)")
    args = ap.parse_args()

    base_seed = args.seed if args.seed is not None else random.randrange(1 << 30)

    with open(args.isa) as f:
        instructions = json.load(f)

    # CT mode: include loads (memory accesses are exactly what CT observes)
    # Exclude control-flow (branches need labels)
    testable = [i for i in instructions if not i.get("is_control_flow", False)]
    if not args.include_loads:
        testable = [i for i in testable if "LOAD" not in i["category"]]

    print(f"CT-SEQ contract checker")
    print(f"Base seed : {base_seed}   runs: {args.runs}   testable: {len(testable)}")
    print("=" * 68)

    n_safe = n_viol = n_err = 0
    violations = []

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

        core = build_core_ct(instr_line)
        code  = compile_core(core)
        if code is None:
            print(f"[{seed}] {instr['name']:7s} {instr_line:35s} ERROR (compile)")
            n_err += 1
            continue

        status, ta, tb, diffs = check_ct(instr_line, code, seed)

        if status == "safe":
            n_safe += 1
            print(f"[{seed}] {instr['name']:7s} {instr_line:35s} CT-safe")
        else:
            n_viol += 1
            violations.append((seed, instr, instr_line, ta, tb, diffs))
            print(f"[{seed}] {instr['name']:7s} {instr_line:35s} <<< VIOLATION")

    # Detail for single run
    if args.runs == 1 and (n_safe + n_viol) == 1:
        seed = base_seed
        random.seed(seed)
        instr = random.choice(testable)
        instr_line = format_instruction(instr)
        core = build_core_ct(instr_line)
        code  = compile_core(core)
        if code:
            state_a, state_b = make_input_pair(None, seed)
            ta = collect_ct_trace(code, state_a)
            tb = collect_ct_trace(code, state_b)
            print("\n--- CT trace for input A ---")
            for ev in ta[:20]:
                print(f"  {ev[0]:3s}  0x{ev[1]:08x}")
            if len(ta) > 20:
                print(f"  ... ({len(ta)} events total)")
            print("\n--- CT trace for input B ---")
            for ev in tb[:20]:
                print(f"  {ev[0]:3s}  0x{ev[1]:08x}")
            if len(tb) > 20:
                print(f"  ... ({len(tb)} events total)")

    print("\n" + "=" * 68)
    print(f"runs={args.runs}  CT-safe={n_safe}  violation={n_viol}  error={n_err}")

    for seed, instr, line, ta, tb, diffs in violations:
        print(f"\n{'─'*68}")
        print(f"VIOLATION  seed={seed}  {instr['name']}  {line}")
        print("  First diverging events:")
        for i, ea, eb in diffs[:5]:
            print(f"    event[{i}]: A=({ea[0]}, 0x{ea[1]:08x})  B=({eb[0]}, 0x{eb[1]:08x})")
        if len(diffs) > 5:
            print(f"    ... ({len(diffs)} diverging events total)")


if __name__ == "__main__":
    main()