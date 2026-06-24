"""
Second Unicorn PoC: same idea as riscv_unicorn_poc.py, with two improvements.

Differences from v1:
  - Operand lookup by name (robust to JSON ordering changes).
  - Random 64-bit initial register values + diff of modified registers
    after execution, so the effect of the test instruction is visible.
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from unicorn import Uc, UC_ARCH_RISCV, UC_MODE_RISCV64, UcError, UC_HOOK_CODE
from unicorn.riscv_const import *


RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")

ADDRESS_CODE = 0x10000
REGION_SIZE  = 2 * 1024 * 1024   # 2 MiB

# x1..x31 — x0 is hardwired to zero so we skip it.
TRACKED_REGS = list(range(1, 32))
UC_REG = {i: globals()[f"UC_RISCV_REG_X{i}"] for i in range(32)}


# ── Operand pickers ──────────────────────────────────────────────────────────

def pick_register(operand):
    return random.choice(operand["values"])


def pick_immediate(operand):
    raw = operand["values"][0]
    m = RANGE_RE.match(raw)
    return random.randint(int(m.group(1)), int(m.group(2)))


def get_operand_by_name(operands, name):
    """Look up an operand by name — robust to JSON ordering changes."""
    for op in operands:
        if op["name"] == name:
            return op
    raise KeyError(f"Operand '{name}' not found")


# ── Instruction formatting ───────────────────────────────────────────────────

def format_instruction(instr):
    """Render one JSON entry as an assembly line, addressing operands by name."""
    name = instr["name"].lower()
    cat  = instr["category"]
    ops  = instr["operands"]

    if cat == "RV64I-LOAD":
        rd     = pick_register(get_operand_by_name(ops, "rd"))
        offset = pick_immediate(get_operand_by_name(ops, "offset"))
        rs1    = pick_register(get_operand_by_name(ops, "rs1"))
        return f"{name} {rd}, {offset}({rs1})"

    if cat == "RV64I-STORE":
        rs2    = pick_register(get_operand_by_name(ops, "rs2"))
        offset = pick_immediate(get_operand_by_name(ops, "offset"))
        rs1    = pick_register(get_operand_by_name(ops, "rs1"))
        return f"{name} {rs2}, {offset}({rs1})"

    if cat == "RV64I-INDIRECT-BR":
        rd  = pick_register(get_operand_by_name(ops, "rd"))
        rs1 = pick_register(get_operand_by_name(ops, "rs1"))
        imm = pick_immediate(get_operand_by_name(ops, "imm12"))
        return f"{name} {rd}, {imm}({rs1})"

    if cat == "RV64I-UPPER-IMM":
        rd  = pick_register(get_operand_by_name(ops, "rd"))
        imm = pick_immediate(get_operand_by_name(ops, "imm20"))
        return f"{name} {rd}, {imm}"

    # Default: positional REG/IMM operands.
    parts = []
    for op in ops:
        if op["type_"] == "REG":
            parts.append(pick_register(op))
        elif op["type_"] == "IMM":
            parts.append(str(pick_immediate(op)))
        else:
            raise ValueError(f"Unexpected operand type: {op['type_']}")
    return f"{name} {', '.join(parts)}"


# ── Compilation via riscv64-elf binutils ─────────────────────────────────────

def compile_to_binary(asm_line):
    """Assemble + objcopy one instruction. Returns (bytes, message) or (None, error)."""
    source = (
        ".section .text\n"
        ".globl _start\n"
        "_start:\n"
        f"    {asm_line}\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        src  = Path(tmp) / "test.s"
        obj  = Path(tmp) / "test.o"
        bin_ = Path(tmp) / "test.bin"
        src.write_text(source, encoding="utf-8")

        # march=rv64im must match UC_MODE_RISCV64 used below.
        r = subprocess.run(
            ["riscv64-elf-as", "-march=rv64im", "-o", str(obj), str(src)],
            capture_output=True,
        )
        if r.returncode != 0:
            return None, f"Assembler error: {r.stderr.decode()}"

        r = subprocess.run(
            ["riscv64-elf-objcopy", "-O", "binary", str(obj), str(bin_)],
            capture_output=True,
        )
        if r.returncode != 0:
            return None, f"objcopy error: {r.stderr.decode()}"

        return bin_.read_bytes(), "OK"


# ── Execution in Unicorn ─────────────────────────────────────────────────────

def run_in_unicorn(machine_code):
    """Execute one instruction with random initial register state. Diff before/after."""
    print("\nStarting Unicorn...")

    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64)
        mu.mem_map(ADDRESS_CODE, REGION_SIZE)
        mu.mem_write(ADDRESS_CODE, machine_code)

        # Random 64-bit initial values for x1..x31.
        initial = {}
        for i in TRACKED_REGS:
            val = random.getrandbits(64)
            mu.reg_write(UC_REG[i], val)
            initial[i] = val

        print("\nInitial register values:")
        for i in TRACKED_REGS:
            print(f"  x{i:<2} = 0x{initial[i]:016x}")

        def hook_code(uc, address, size, user_data):
            print(f"\n  -> executing at 0x{address:x}")
        mu.hook_add(UC_HOOK_CODE, hook_code)

        mu.emu_start(ADDRESS_CODE, ADDRESS_CODE + len(machine_code))

        # Diff: only show registers whose value changed.
        changed = []
        for i in TRACKED_REGS:
            new_val = mu.reg_read(UC_REG[i])
            if new_val != initial[i]:
                changed.append((i, initial[i], new_val))

        if changed:
            print("\nChanged registers:")
            for i, old, new in changed:
                print(f"  x{i:<2} : 0x{old:016x}  ->  0x{new:016x}")
        else:
            print("\n(no registers changed)")

        return True

    except UcError as e:
        print(f"Unicorn error: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Unicorn PoC v2 — named operands + register diff")
    ap.add_argument("--json", default="riscv_isa.json", help="Path to the ISA spec")
    ap.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"1. Loading {args.json}...")
    with open(args.json, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    # Exclude:
    #  - control flow (no real branch target, would jump to unmapped memory)
    #  - loads/stores (random 64-bit registers would point outside mapped region)
    #  - *W variants (RV64-only sign-extension semantics, not the focus here)
    testable = [
        i for i in instructions
        if not i.get("is_control_flow", False)
        and i["category"] not in ("RV64I-LOAD", "RV64I-STORE")
        and not i["name"].endswith("W")
    ]

    instr    = random.choice(testable)
    asm_line = format_instruction(instr)

    print(f"2. Generated instruction: {asm_line}")
    print("3. Compiling to raw bytecode...")
    code, msg = compile_to_binary(asm_line)

    if code is None:
        print(msg)
        return 1

    print(f"   raw bytes: {code.hex()}")
    run_in_unicorn(code)
    return 0


if __name__ == "__main__":
    sys.exit(main())