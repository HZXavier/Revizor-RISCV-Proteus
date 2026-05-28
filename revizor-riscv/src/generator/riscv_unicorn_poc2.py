"""
Full RISC-V pipeline (corrected version).

For a randomly picked instruction from the JSON:
  1. Pick operands,
  2. Build the assembly text,
  3. Assemble + extract bytecode,
  4. Run in Unicorn and print registers.

Fixes over the previous version:
  - Correct category names: RV64I-LOAD and RV64I-STORE
  - Valid RISC-V syntax: offset(rs1) instead of offset, rs1
  - march=rv64im to match UC_MODE_RISCV64
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


def pick_register(operand):
    return random.choice(operand["values"])


def pick_immediate(operand):
    raw = operand["values"][0]
    m = RANGE_RE.match(raw)
    return random.randint(int(m.group(1)), int(m.group(2)))


def get_operand_by_name(operands, name):
    """Look up an operand by name. More robust than relying on list index."""
    for op in operands:
        if op["name"] == name:
            return op
    raise KeyError(f"Operand '{name}' not found")


def format_instruction(instr):
    """
    Build the assembly line by category and operand name.
    Using names instead of indices means the code stays correct
    even if the JSON operand order changes.
    """
    name = instr["name"].lower()
    cat = instr["category"]
    ops = instr["operands"]

    # Loads: LX rd, offset(rs1)
    if cat == "RV64I-LOAD":
        rd     = pick_register(get_operand_by_name(ops, "rd"))
        offset = pick_immediate(get_operand_by_name(ops, "offset"))
        rs1    = pick_register(get_operand_by_name(ops, "rs1"))
        return f"{name} {rd}, {offset}({rs1})"

    # Stores: SX rs2, offset(rs1)
    if cat == "RV64I-STORE":
        rs2    = pick_register(get_operand_by_name(ops, "rs2"))
        offset = pick_immediate(get_operand_by_name(ops, "offset"))
        rs1    = pick_register(get_operand_by_name(ops, "rs1"))
        return f"{name} {rs2}, {offset}({rs1})"

    # JALR: jalr rd, imm12(rs1)
    if cat == "RV64I-INDIRECT-BR":
        rd  = pick_register(get_operand_by_name(ops, "rd"))
        rs1 = pick_register(get_operand_by_name(ops, "rs1"))
        imm = pick_immediate(get_operand_by_name(ops, "imm12"))
        return f"{name} {rd}, {imm}({rs1})"

    # LUI / AUIPC: OP rd, imm20
    if cat == "RV64I-UPPER-IMM":
        rd  = pick_register(get_operand_by_name(ops, "rd"))
        imm = pick_immediate(get_operand_by_name(ops, "imm20"))
        return f"{name} {rd}, {imm}"

    # Default: OP arg1, arg2, arg3 (R-type, I-type, shifts, M-ext)
    parts = []
    for op in ops:
        if op["type_"] == "REG":
            parts.append(pick_register(op))
        elif op["type_"] == "IMM":
            parts.append(str(pick_immediate(op)))
        else:
            raise ValueError(f"Unexpected operand type: {op['type_']}")
    return f"{name} {', '.join(parts)}"


def compile_to_binary(asm_line):
    """Assemble the line and return (bytecode, message)."""
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

        # Step 1: text -> object file
        # march=rv64im must match UC_MODE_RISCV64 used in Unicorn
        r_as = subprocess.run(
            ["riscv64-elf-as", "-march=rv64im", "-o", str(obj_path), str(src_path)],
            capture_output=True,
        )
        if r_as.returncode != 0:
            return None, f"Assembler error: {r_as.stderr.decode()}"

        # Step 2: object -> raw bytecode
        r_oc = subprocess.run(
            ["riscv64-elf-objcopy", "-O", "binary", str(obj_path), str(bin_path)],
            capture_output=True,
        )
        if r_oc.returncode != 0:
            return None, f"Objcopy error: {r_oc.stderr.decode()}"

        return bin_path.read_bytes(), "OK"


def run_in_unicorn(machine_code):
    print("\nStarting Unicorn...")

    ADDRESS_CODE = 0x10000

    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64)
        mu.mem_map(ADDRESS_CODE, 2 * 1024 * 1024)
        mu.mem_write(ADDRESS_CODE, machine_code)

        regs = [getattr(sys.modules[__name__], f"UC_RISCV_REG_X{i}") for i in range(1, 32)]

        # Give each register a random 64-bit value (x0 is hardwired zero, cannot be written)
        initial_values = {}
        for i, reg in enumerate(regs, start=1):
            val = random.getrandbits(64)
            mu.reg_write(reg, val)
            initial_values[f"x{i}"] = val

        print("\nInitial register values:")
        for name, val in initial_values.items():
            print(f"  {name:4s} = 0x{val:016x}")

        def hook_code(uc, address, size, user_data):
            print(f"\n -> Executing at 0x{address:x}")
        mu.hook_add(UC_HOOK_CODE, hook_code)

        mu.emu_start(ADDRESS_CODE, ADDRESS_CODE + len(machine_code))

        # Show which registers changed
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

        return True

    except UcError as e:
        print(f"Unicorn error: {e}")
        return False


if __name__ == "__main__":
    print("1. Loading JSON instruction dictionary...")
    with open("riscv_isa.json", "r") as f:
        instructions = json.load(f)

    # Skip branches (would jump to unmapped memory) and *W variants for now
    instrs_simples = [
        i for i in instructions
        if not i.get("is_control_flow", False)
        and i["category"] not in ("RV64I-LOAD", "RV64I-STORE")
        and not i["name"].endswith("W")
    ]

    instr = random.choice(instrs_simples)
    asm_line = format_instruction(instr)

    print(f"2. Generated instruction: {asm_line}")
    print("3. Compiling and extracting bytecode...")

    machine_code, msg = compile_to_binary(asm_line)

    if machine_code:
        print(f" -> Raw binary: {machine_code.hex()}")
        run_in_unicorn(machine_code)
    else:
        print(msg)
