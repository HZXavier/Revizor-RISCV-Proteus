"""
Runs a random RV32IM instruction on the Proteus simulator via Docker.

Pipeline:
  1. Pick a random instruction from riscv_isa_rv32.json
  2. Generate assembly with a RV32 prologue/epilogue
  3. Cross-compile inside Docker (riscv32-unknown-elf-gcc)
  4. Run on Proteus simulator inside Docker
  5. Parse and print the register dump
"""
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")

DOCKER_IMAGE  = "proteus"
SANDBOX_ADDR  = 0x80001000          # sandbox base in Proteus RAM (starts at 0x80000000)
RESERVED_REGS = {"x30", "x31"}      # never clobbered: sandbox base + aux


def pick_register(operand, exclude=None):
    candidates = [v for v in operand["values"] if v not in (exclude or set())]
    if not candidates:
        raise ValueError(f"No registers left after excluding {exclude}")
    return random.choice(candidates)


def pick_immediate(operand):
    raw = operand["values"][0]
    m = RANGE_RE.match(raw)
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
        rd     = pick_register(get_op(ops, "rd"), exclude=RESERVED_REGS)
        offset = pick_immediate(get_op(ops, "offset"))
        return f"{name} {rd}, {offset}(x30)"

    if "STORE" in cat:
        rs2    = pick_register(get_op(ops, "rs2"))
        offset = pick_immediate(get_op(ops, "offset"))
        return f"{name} {rs2}, {offset}(x30)"

    if "INDIRECT-BR" in cat:
        rd  = pick_register(get_op(ops, "rd"))
        rs1 = pick_register(get_op(ops, "rs1"))
        imm = pick_immediate(get_op(ops, "imm12"))
        return f"{name} {rd}, {imm}({rs1})"

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


def build_asm_source(instr_line):
    """Minimal RV32 program: init sandbox + test instruction + reg dump + EOT halt."""
    hi = (SANDBOX_ADDR >> 12) & 0xFFFFF
    lo = SANDBOX_ADDR & 0xFFF
    return f""".section .text
.globl _start
_start:
    # sandbox base -> x30
    lui   x30, {hi}
    addi  x30, x30, {lo}

    # init general registers with small known values
    addi  x5,  x0, 1
    addi  x6,  x0, 2
    addi  x7,  x0, 3
    addi  x8,  x0, 4
    addi  x9,  x0, 5
    addi  x10, x0, 6
    addi  x11, x0, 7
    addi  x12, x0, 8
    addi  x13, x0, 9
    addi  x14, x0, 10
    addi  x15, x0, 11

    # === TEST INSTRUCTION ===
    {instr_line}
    # ========================

    # dump registers — x31 comme pointeur (réservé)
    lui  x31, 0x80002
    sw   x5,  0(x31)
    sw   x6,  4(x31)
    sw   x7,  8(x1)
    sw   x8,  12(x1)
    sw   x9,  16(x1)
    sw   x10, 20(x1)
    sw   x11, 24(x1)
    sw   x12, 28(x1)
    sw   x13, 32(x1)
    sw   x14, 36(x1)
    sw   x15, 40(x1)
    sw   x16, 44(x1)
    sw   x17, 48(x1)
    sw   x18, 52(x1)
    sw   x19, 56(x1)
    sw   x20, 60(x1)
    sw   x21, 64(x1)
    sw   x22, 68(x1)
    sw   x23, 72(x1)
    sw   x24, 76(x1)
    sw   x25, 80(x1)
    sw   x26, 84(x1)
    sw   x27, 88(x1)
    sw   x28, 92(x1)
    sw   x29, 96(x31)

    # halt via CharDev — x30/x31 réservés
    lui  x30, 0x10000
    li   x31, 4
    sb   x31, 0(x30)
1:  j 1b
"""


def run_in_docker(asm_source):
    """Compile + simulate in Docker. Returns (output_str, success_bool)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "test.S"
        src.write_text(asm_source)

        # 1. Compile
        r = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{tmpdir}:/work",
            DOCKER_IMAGE,
            "riscv32-unknown-elf-gcc",
            "-march=rv32im_zicsr", "-mabi=ilp32",
            "-ffreestanding", "-nostdlib",
            "-Wl,-Ttext=0x80000000",
            "-o", "/work/test.elf",
            "/work/test.S",
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return r.stderr, False

        # 2. Extract raw binary
        r = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{tmpdir}:/work",
            DOCKER_IMAGE,
            "riscv32-unknown-elf-objcopy",
            "-O", "binary",
            "/work/test.elf", "/work/test.bin",
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return r.stderr, False

        # 3. Simulate on Proteus
        r = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{tmpdir}:/work",
            DOCKER_IMAGE,
            "/proteus/sim/build/sim", "/work/test.bin",
        ], capture_output=True, text=True, timeout=30)

        return r.stdout + r.stderr, (r.returncode == 0)


def print_output(output, ok):
    """Parse and display Proteus output, highlighting the register dump."""
    status = "[OK]" if ok else "[KO]"
    print(f"\n{status} Proteus output:")

    if not output.strip():
        print("  (no output)")
        return

    if "[REGDUMP]" in output and "[/REGDUMP]" in output:
        before = output[:output.index("[REGDUMP]")].strip()
        after  = output[output.index("[/REGDUMP]") + len("[/REGDUMP]"):].strip()
        dump   = output[output.index("[REGDUMP]") + len("[REGDUMP]"):output.index("[/REGDUMP]")]

        if before:
            print(before)

        print("\nRegister state after execution (Proteus):")
        for line in dump.splitlines():
            line = line.strip()
            if "=" in line:
                reg, val = line.split(" = ")
                print(f"  {reg} = {val}")

        if after:
            print(f"\n{after}")
    else:
        # REGDUMP not found: image not yet rebuilt with new main.cpp
        print(output)
        print("\n[!] No REGDUMP found — rebuild the Docker image with the updated main.cpp")


def main():
    print("Loading RV32IM ISA...")
    with open("riscv_isa_rv32.json") as f:
        instructions = json.load(f)

    # Skip control-flow instructions (branches need labels not present in this harness)
    testable = [i for i in instructions if not i.get("is_control_flow", False)]

    instr      = random.choice(testable)
    instr_line = format_instruction(instr)

    print(f"Instruction : {instr['name']}")
    print(f"Assembly    : {instr_line}")
    print(f"Category    : {instr['category']}")

    asm = build_asm_source(instr_line)
    print("\n--- Assembly source ---")
    print(asm)
    print("-----------------------")

    print("Running on Proteus (Docker)...")
    output, ok = run_in_docker(asm)
    print_output(output, ok)


if __name__ == "__main__":
    main()