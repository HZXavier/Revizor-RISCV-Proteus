"""
Early proof-of-concept: assemble one random RISC-V instruction and run it on Unicorn.

This was the first integration test of the project, before Proteus and the
full RV32 pipeline were in place
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
ADDRESS_DATA = 0x300000
REGION_SIZE  = 2 * 1024 * 1024   # 2 MiB per region

# x1..x31 — x0 is hardwired to zero so we skip it.
TRACKED_REGS = list(range(1, 32))
UC_REG = {i: globals()[f"UC_RISCV_REG_X{i}"] for i in range(32)}


# ── Instruction generation (same conventions as create_instructions.py) ──────

def pick_register(operand):
    return random.choice(operand["values"])


def pick_immediate(operand):
    raw = operand["values"][0]
    m = RANGE_RE.match(raw)
    return random.randint(int(m.group(1)), int(m.group(2)))


def format_instruction(instr):
    """Build the assembly line from one JSON entry."""
    name = instr["name"].lower()
    cat  = instr["category"]
    ops  = instr["operands"]

    # Loads: LX rd, offset(rs1)
    if "LOAD" in cat:
        return f"{name} {pick_register(ops[0])}, {pick_immediate(ops[1])}({pick_register(ops[2])})"

    # Stores: SX rs2, offset(rs1)
    if "STORE" in cat:
        return f"{name} {pick_register(ops[0])}, {pick_immediate(ops[1])}({pick_register(ops[2])})"

    # LUI / AUIPC: OP rd, imm20
    if name in ("lui", "auipc"):
        return f"{name} {pick_register(ops[0])}, {pick_immediate(ops[1])}"

    # Default: positional REG/IMM operands.
    parts = []
    for op in ops:
        if op["type_"] == "REG":
            parts.append(pick_register(op))
        elif op["type_"] == "IMM":
            parts.append(str(pick_immediate(op)))
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
        src = Path(tmp) / "test.s"
        obj = Path(tmp) / "test.o"
        bin_ = Path(tmp) / "test.bin"
        src.write_text(source, encoding="utf-8")

        r = subprocess.run(
            ["riscv64-elf-as", "-march=rv32im", "-o", str(obj), str(src)],
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
    """Execute one instruction in Unicorn (RISCV64 mode) and print register state."""
    print("\nStarting Unicorn...")

    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64)

        # Separate code and data regions to keep memory mapping simple.
        mu.mem_map(ADDRESS_CODE, REGION_SIZE)
        mu.mem_map(ADDRESS_DATA, REGION_SIZE)
        mu.mem_write(ADDRESS_CODE, machine_code)

        # Point every register at a valid data address so loads/stores don't fault.
        for i in TRACKED_REGS:
            mu.reg_write(UC_REG[i], ADDRESS_DATA + 0x100)

        def hook_code(uc, address, size, user_data):
            print(f"  -> executing at 0x{address:x}")
        mu.hook_add(UC_HOOK_CODE, hook_code)

        mu.emu_start(ADDRESS_CODE, ADDRESS_CODE + len(machine_code))

        print("\nFinal register state:")
        for i in TRACKED_REGS:
            val = mu.reg_read(UC_REG[i])
            print(f"  x{i:<2} = {val:>20}  (0x{val:016x})")

        return True

    except UcError as e:
        print(f"Unicorn error: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Early Unicorn PoC for RV32IM execution")
    ap.add_argument("--json", default="riscv_isa.json", help="Path to the ISA spec")
    ap.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"1. Loading {args.json}...")
    with open(args.json, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    # Keep simple data-path instructions: no control flow, no RV64-only (*W, LD, SD).
    simple = [
        i for i in instructions
        if not i.get("is_control_flow", False)
        and not i["name"].endswith("W")
        and i["name"] not in ("LD", "SD")
    ]

    instr    = random.choice(simple)
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