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


def pick_register(operand):
    return random.choice(operand["values"])

def pick_immediate(operand):
    raw = operand["values"][0]
    m = RANGE_RE.match(raw)
    return random.randint(int(m.group(1)), int(m.group(2)))

def format_instruction(instr):
    """Build the assembly line from the JSON instruction data."""
    name = instr["name"].lower()
    cat = instr["category"]
    operands = instr["operands"]

    # Memory access: e.g. lw x5, 100(x6)
    if cat == "BASE-DATAXFER":
        is_store = name.startswith('s')
        reg1 = pick_register(operands[0])
        reg2 = pick_register(operands[1])
        offset = pick_immediate(operands[2])
        if is_store:
            return f"{name} {reg2}, {offset}({reg1})"
        else:
            return f"{name} {reg1}, {offset}({reg2})"

    # Upper immediate: e.g. lui x5, 4000
    if name in ["lui", "auipc"]:
        return f"{name} {pick_register(operands[0])}, {pick_immediate(operands[1])}"

    # Default: e.g. add x5, x6, x7
    parts = []
    for op in operands:
        if op["type_"] == "REG":
            parts.append(pick_register(op))
        elif op["type_"] == "IMM":
            parts.append(str(pick_immediate(op)))
    return f"{name} {', '.join(parts)}"


def compile_to_binary(asm_line):
    source = f".section .text\n.globl _start\n_start:\n    {asm_line}\n"

    # On écrit directement dans le dossier courant (fini le tempfile)
    src_path = Path("test.s")
    obj_path = Path("test.o")
    bin_path = Path("test.bin")

    src_path.write_text(source, encoding="utf-8")

    # Step 1: assemble text into object file
    res_as = subprocess.run(
        ["riscv64-elf-as", "-march=rv32im", "-o", str(obj_path), str(src_path)],
        capture_output=True,
    )
    if res_as.returncode != 0:
        return None, f"Assembler error: {res_as.stderr.decode()}"

    # Step 2: extract raw machine code from object file
    res_obj = subprocess.run(
        ["riscv64-elf-objcopy", "-O", "binary", str(obj_path), str(bin_path)],
        capture_output=True,
    )
    if res_obj.returncode != 0:
        return None, f"Objcopy error: {res_obj.stderr.decode()}"

    return bin_path.read_bytes(), "OK"


def run_in_unicorn(machine_code):
    print("\nStarting Unicorn...")
    ADDRESS_CODE = 0x10000
    ADDRESS_DATA = 0x300000

    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64)

        # Map separate regions for code and data to avoid UC_ERR_MAP
        mu.mem_map(ADDRESS_CODE, 2 * 1024 * 1024)
        mu.mem_map(ADDRESS_DATA, 2 * 1024 * 1024)

        mu.mem_write(ADDRESS_CODE, machine_code)

        regs = [getattr(sys.modules[__name__], f"UC_RISCV_REG_X{i}") for i in range(1, 32)]

        # Point all registers at a valid data address to avoid memory faults
        for reg in regs:
            mu.reg_write(reg, ADDRESS_DATA + 0x100)

        def hook_code(uc, address, size, user_data):
            print(f" -> Executing at 0x{address:x}")
        mu.hook_add(UC_HOOK_CODE, hook_code)

        mu.emu_start(ADDRESS_CODE, ADDRESS_CODE + len(machine_code))

        print("\nExecution done. Register state:")
        for i, reg in enumerate(regs):
            val = mu.reg_read(reg)
            print(f" x{i+1} = {val} (0x{val:08x})")

        return True

    except UcError as e:
        print(f"Unicorn error: {e}")
        return False


if __name__ == "__main__":
    print("1. Loading JSON instruction dictionary...")
    with open("riscv_isa.json", "r") as f:
        instructions = json.load(f)

    # Skip branches and 64-bit-only instructions (W, LD, SD)
    instrs_simples = [
        i for i in instructions
        if not i.get("is_control_flow", False)
        and not i["name"].endswith("W")
        and i["name"] not in ["LD", "SD"]
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
