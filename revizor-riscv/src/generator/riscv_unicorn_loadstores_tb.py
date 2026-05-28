"""
Full RISC-V pipeline with memory sandbox.

For a randomly picked instruction from the JSON:
  1. Pick operands (with sandbox constraint for loads/stores),
  2. Build the assembly text,
  3. Assemble + extract bytecode,
  4. Run in Unicorn and print registers.

Memory sandbox:
  - x30 = base of the DATA region (0x300000), used as base address for all loads/stores
  - x31 = reserved auxiliary register
  - All other registers get random values
"""

import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from unicorn import Uc, UC_ARCH_RISCV, UC_MODE_RISCV64, UcError, UC_HOOK_CODE
from unicorn.riscv_const import *  # noqa: imports UC_RISCV_REG_X*


RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")

ADDRESS_CODE = 0x10000   # region where emulated code lives
ADDRESS_DATA = 0x300000  # region where loads/stores go

# x30 is set to ADDRESS_DATA so any address x30+offset stays inside the mapped region
# as long as offset stays within [-2048, 2047]
SANDBOX_BASE_REG = "x30"
SANDBOX_AUX_REG  = "x31"

# These registers must not be overwritten by random instructions,
# otherwise the sandbox breaks for any load/store that follows
RESERVED_REGS = {SANDBOX_BASE_REG, SANDBOX_AUX_REG}


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


def format_instruction(instr):
    """Build the assembly line. Forces rs1=x30 for loads/stores to stay in the sandbox."""
    name = instr["name"].lower()
    cat = instr["category"]
    ops = instr["operands"]

    # Loads: LX rd, offset(rs1) — force rs1=x30, protect rd from clobbering sandbox regs
    if cat == "RV64I-LOAD":
        rd     = pick_register(get_operand_by_name(ops, "rd"), exclude=RESERVED_REGS)
        offset = pick_immediate(get_operand_by_name(ops, "offset"))
        rs1    = SANDBOX_BASE_REG
        return f"{name} {rd}, {offset}({rs1})"

    # Stores: SX rs2, offset(rs1) — force rs1=x30, rs2 can be anything
    if cat == "RV64I-STORE":
        rs2    = pick_register(get_operand_by_name(ops, "rs2"))
        offset = pick_immediate(get_operand_by_name(ops, "offset"))
        rs1    = SANDBOX_BASE_REG
        return f"{name} {rs2}, {offset}({rs1})"

    # JALR: jalr rd, imm12(rs1)
    if cat == "RV64I-INDIRECT-BR":
        rd  = pick_register(get_operand_by_name(ops, "rd"))
        rs1 = pick_register(get_operand_by_name(ops, "rs1"))
        imm = pick_immediate(get_operand_by_name(ops, "imm12"))
        return f"{name} {rd}, {imm}({rs1})"

    # LUI / AUIPC: OP rd, imm20 — protect rd from clobbering sandbox regs
    if cat == "RV64I-UPPER-IMM":
        rd  = pick_register(get_operand_by_name(ops, "rd"), exclude=RESERVED_REGS)
        imm = pick_immediate(get_operand_by_name(ops, "imm20"))
        return f"{name} {rd}, {imm}"

    # Default: OP arg1, arg2, arg3 — protect destination from clobbering sandbox regs
    parts = []
    for op in ops:
        if op["type_"] == "REG":
            if op["dest"]:
                parts.append(pick_register(op, exclude=RESERVED_REGS))
            else:
                parts.append(pick_register(op))
        elif op["type_"] == "IMM":
            parts.append(str(pick_immediate(op)))
        else:
            raise ValueError(f"Unexpected operand type: {op['type_']}")
    return f"{name} {', '.join(parts)}"


def compile_to_binary(asm_line):
    source = (
        ".section .text\n"
        ".globl _start\n"
        "_start:\n"
        f"    {asm_line}\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "test.s"
        obj_path = Path(tmpdir) / "test.o"
        bin_path = Path(tmpdir) / "test.bin"
        src_path.write_text(source, encoding="utf-8")

        r_as = subprocess.run(
            ["riscv64-elf-as", "-march=rv64im", "-o", str(obj_path), str(src_path)],
            capture_output=True,
        )
        if r_as.returncode != 0:
            return None, f"Assembler error: {r_as.stderr.decode()}"

        r_oc = subprocess.run(
            ["riscv64-elf-objcopy", "-O", "binary", str(obj_path), str(bin_path)],
            capture_output=True,
        )
        if r_oc.returncode != 0:
            return None, f"Objcopy error: {r_oc.stderr.decode()}"

        return bin_path.read_bytes(), "OK"


def run_in_unicorn(machine_code, instr):
    """Run bytecode in Unicorn with a DATA memory region and sandbox registers."""
    print("\nStarting Unicorn...")

    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64)

        # Map CODE region: 2 MB at ADDRESS_CODE
        mu.mem_map(ADDRESS_CODE, 2 * 1024 * 1024)

        # Map DATA region: 1 MB starting 64 KB before ADDRESS_DATA
        # so that x30 + a negative offset still lands inside the mapped area
        DATA_PAGE_START = ADDRESS_DATA - 0x10000
        DATA_PAGE_SIZE  = 1 * 1024 * 1024
        mu.mem_map(DATA_PAGE_START, DATA_PAGE_SIZE)

        mu.mem_write(ADDRESS_CODE, machine_code)

        # Fill DATA with a recognizable pattern so loads return non-zero values
        pattern = b"\xAB\xCD\xEF\x01" * (DATA_PAGE_SIZE // 4)
        mu.mem_write(DATA_PAGE_START, pattern)

        regs = [getattr(sys.modules[__name__], f"UC_RISCV_REG_X{i}") for i in range(1, 32)]

        initial_values = {}
        for i, reg in enumerate(regs, start=1):
            name = f"x{i}"
            if name == SANDBOX_BASE_REG:
                val = ADDRESS_DATA
            elif name == SANDBOX_AUX_REG:
                val = 0x0000_0000_FFFF_FFFF
            else:
                val = random.getrandbits(64)
            mu.reg_write(reg, val)
            initial_values[name] = val

        print("\nInitial register values:")
        for name, val in initial_values.items():
            marker = "  (sandbox)" if name in RESERVED_REGS else ""
            print(f"  {name:4s} = 0x{val:016x}{marker}")

        def hook_code(uc, address, size, user_data):
            print(f"\n -> Executing at 0x{address:x}")
        mu.hook_add(UC_HOOK_CODE, hook_code)

        mu.emu_start(ADDRESS_CODE, ADDRESS_CODE + len(machine_code))

        print("\nRegister state after execution:")
        changed = []
        for i, reg in enumerate(regs, start=1):
            name = f"x{i}"
            new_val = mu.reg_read(reg)
            old_val = initial_values[name]
            if new_val != old_val:
                changed.append((name, old_val, new_val))

        if changed:
            print("\nChanged registers:")
            for name, old, new in changed:
                print(f"  {name:4s} : 0x{old:016x}  ->  0x{new:016x}")
        else:
            print("  (no registers changed)")

        # For stores, dump memory near the sandbox base to confirm the write landed
        if instr["category"] == "RV64I-STORE":
            print("\nMemory near x30 (sandbox base):")
            dump = mu.mem_read(ADDRESS_DATA - 16, 64)
            for off in range(0, 64, 8):
                addr = ADDRESS_DATA - 16 + off
                octets = dump[off:off+8]
                print(f"  [0x{addr:08x}] {octets.hex()}")

        return True

    except UcError as e:
        print(f"Unicorn error: {e}")
        return False


if __name__ == "__main__":
    print("1. Loading JSON instruction dictionary...")
    with open("riscv_isa.json", "r") as f:
        instructions = json.load(f)

    # Keep loads and stores, skip branches (would jump to unmapped memory) and *W variants
    instrs_simples = [
        i for i in instructions
        if not i.get("is_control_flow", False)
        and not i["name"].endswith("W")
    ]

    instr = random.choice(instrs_simples)
    asm_line = format_instruction(instr)

    print(f"2. Generated instruction: {asm_line}")
    print("3. Compiling and extracting bytecode...")

    machine_code, msg = compile_to_binary(asm_line)

    if machine_code:
        print(f" -> Raw binary: {machine_code.hex()}")
        run_in_unicorn(machine_code, instr)
    else:
        print(msg)
