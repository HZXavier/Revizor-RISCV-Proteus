"""
Third Unicorn PoC: introduces a memory sandbox so loads and stores actually work.

Difference from v2:
  - x30 is reserved as the base address of a mapped DATA region.
  - Loads and stores are forced to use x30 as the base register, so any
    offset in [-2048, 2047] lands inside the sandbox.
  - DATA region is pre-filled with a recognizable pattern so loads return
    non-zero, observable values.
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

# Map the DATA region one page below ADDRESS_DATA so x30 + negative offsets
# still land inside the mapped area.
DATA_PAGE_START = ADDRESS_DATA - 0x10000
DATA_PAGE_SIZE  = 1 * 1024 * 1024   # 1 MiB
CODE_REGION_SIZE = 2 * 1024 * 1024  # 2 MiB

# Sandbox base register: holds ADDRESS_DATA, used as rs1 for every load/store.
SANDBOX_BASE_REG = "x30"
RESERVED_REGS    = {SANDBOX_BASE_REG}

# x1..x31 — x0 is hardwired to zero so we skip it.
TRACKED_REGS = list(range(1, 32))
UC_REG = {i: globals()[f"UC_RISCV_REG_X{i}"] for i in range(32)}


# ── Operand pickers ──────────────────────────────────────────────────────────

def pick_register(operand, exclude=None):
    candidates = operand["values"]
    if exclude:
        candidates = [r for r in candidates if r not in exclude]
    if not candidates:
        raise ValueError(f"No registers left after excluding {exclude}")
    return random.choice(candidates)


def pick_immediate(operand):
    raw = operand["values"][0]
    m = RANGE_RE.match(raw)
    return random.randint(int(m.group(1)), int(m.group(2)))


def get_operand_by_name(operands, name):
    for op in operands:
        if op["name"] == name:
            return op
    raise KeyError(f"Operand '{name}' not found")


# ── Instruction formatting ───────────────────────────────────────────────────

def format_instruction(instr):
    """Render one JSON entry as assembly. Forces rs1=x30 for loads/stores."""
    name = instr["name"].lower()
    cat  = instr["category"]
    ops  = instr["operands"]

    # Loads: force rs1=x30, protect rd from clobbering the sandbox base.
    if cat == "RV64I-LOAD":
        rd     = pick_register(get_operand_by_name(ops, "rd"), exclude=RESERVED_REGS)
        offset = pick_immediate(get_operand_by_name(ops, "offset"))
        return f"{name} {rd}, {offset}({SANDBOX_BASE_REG})"

    # Stores: force rs1=x30, rs2 can be anything.
    if cat == "RV64I-STORE":
        rs2    = pick_register(get_operand_by_name(ops, "rs2"))
        offset = pick_immediate(get_operand_by_name(ops, "offset"))
        return f"{name} {rs2}, {offset}({SANDBOX_BASE_REG})"

    if cat == "RV64I-INDIRECT-BR":
        rd  = pick_register(get_operand_by_name(ops, "rd"))
        rs1 = pick_register(get_operand_by_name(ops, "rs1"))
        imm = pick_immediate(get_operand_by_name(ops, "imm12"))
        return f"{name} {rd}, {imm}({rs1})"

    if cat == "RV64I-UPPER-IMM":
        rd  = pick_register(get_operand_by_name(ops, "rd"), exclude=RESERVED_REGS)
        imm = pick_immediate(get_operand_by_name(ops, "imm20"))
        return f"{name} {rd}, {imm}"

    # Default: positional REG/IMM operands. Protect destinations only.
    parts = []
    for op in ops:
        if op["type_"] == "REG":
            excl = RESERVED_REGS if op["dest"] else None
            parts.append(pick_register(op, exclude=excl))
        elif op["type_"] == "IMM":
            parts.append(str(pick_immediate(op)))
        else:
            raise ValueError(f"Unexpected operand type: {op['type_']}")
    return f"{name} {', '.join(parts)}"


# ── Compilation via riscv64-elf binutils ─────────────────────────────────────

def compile_to_binary(asm_line):
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

def run_in_unicorn(machine_code, dump_memory=False):
    """Execute one instruction with a sandboxed DATA region and random GPRs."""
    print("\nStarting Unicorn...")

    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64)

        # Two regions: code and data. The data region is shifted one page below
        # ADDRESS_DATA so negative offsets relative to x30 stay mapped.
        mu.mem_map(ADDRESS_CODE,      CODE_REGION_SIZE)
        mu.mem_map(DATA_PAGE_START,   DATA_PAGE_SIZE)
        mu.mem_write(ADDRESS_CODE, machine_code)

        # Pre-fill data region with a recognizable pattern so loads return non-zero.
        mu.mem_write(DATA_PAGE_START, b"\xAB\xCD\xEF\x01" * (DATA_PAGE_SIZE // 4))

        # Initial register state: x30 = sandbox base, all others random.
        initial = {}
        for i in TRACKED_REGS:
            name = f"x{i}"
            val = ADDRESS_DATA if name == SANDBOX_BASE_REG else random.getrandbits(64)
            mu.reg_write(UC_REG[i], val)
            initial[i] = val

        print("\nInitial register values:")
        for i in TRACKED_REGS:
            tag = "  (sandbox base)" if f"x{i}" == SANDBOX_BASE_REG else ""
            print(f"  x{i:<2} = 0x{initial[i]:016x}{tag}")

        def hook_code(uc, address, size, user_data):
            print(f"\n  -> executing at 0x{address:x}")
        mu.hook_add(UC_HOOK_CODE, hook_code)

        mu.emu_start(ADDRESS_CODE, ADDRESS_CODE + len(machine_code))

        # Show only registers that changed.
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

        # For stores, dump memory near the sandbox base to confirm the write.
        if dump_memory:
            print("\nMemory near sandbox base:")
            base = ADDRESS_DATA - 16
            data = mu.mem_read(base, 64)
            for off in range(0, 64, 8):
                print(f"  [0x{base + off:08x}] {data[off:off+8].hex()}")

        return True

    except UcError as e:
        print(f"Unicorn error: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Unicorn PoC v3 — memory sandbox for loads/stores")
    ap.add_argument("--json", default="riscv_isa.json", help="Path to the ISA spec")
    ap.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"1. Loading {args.json}...")
    with open(args.json, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    # Keep loads and stores (sandbox handles them now), skip control flow and *W.
    testable = [
        i for i in instructions
        if not i.get("is_control_flow", False)
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
    run_in_unicorn(code, dump_memory=(instr["category"] == "RV64I-STORE"))
    return 0


if __name__ == "__main__":
    sys.exit(main())