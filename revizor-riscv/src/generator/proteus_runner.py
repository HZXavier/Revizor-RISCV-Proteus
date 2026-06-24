"""
Runs a single random RV32IM instruction on the Proteus simulator via Docker.
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")

DOCKER_IMAGE  = "proteus"
SANDBOX_ADDR  = 0x80001000      # sandbox base, held in x30
RESERVED_REGS = {"x30", "x31"} # x30 = sandbox base, x31 = dump pointer


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
        rd     = pick_register(get_op(ops, "rd"), exclude=RESERVED_REGS)
        offset = pick_immediate(get_op(ops, "offset"))
        return f"{name} {rd}, {offset}(x30)"

    if "STORE" in cat:
        rs2    = pick_register(get_op(ops, "rs2"))
        offset = pick_immediate(get_op(ops, "offset"))
        if   name == "sw": offset = (offset // 4) * 4
        elif name == "sh": offset = (offset // 2) * 2
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

def build_asm_source(instr_line):
    """Wrap one instruction in a minimal RV32 program."""
    hi = (SANDBOX_ADDR >> 12) & 0xFFFFF
    lo = SANDBOX_ADDR & 0xFFF
    return (
        ".section .text\n"
        ".globl _start\n"
        "_start:\n"
        f"    lui   x30, {hi}\n"
        f"    addi  x30, x30, {lo}\n"
        "\n"
        "    # x5..x15 = 1..11; x16..x29 = 0 (Proteus default)\n"
        "    addi  x5,  x0, 1\n"
        "    addi  x6,  x0, 2\n"
        "    addi  x7,  x0, 3\n"
        "    addi  x8,  x0, 4\n"
        "    addi  x9,  x0, 5\n"
        "    addi  x10, x0, 6\n"
        "    addi  x11, x0, 7\n"
        "    addi  x12, x0, 8\n"
        "    addi  x13, x0, 9\n"
        "    addi  x14, x0, 10\n"
        "    addi  x15, x0, 11\n"
        "\n"
        f"    {instr_line}\n"
        "\n"
        "    # dump x5..x29 to 0x80002000 via x31 (reserved)\n"
        "    lui  x31, 0x80002\n"
        "    sw   x5,   0(x31)\n"
        "    sw   x6,   4(x31)\n"
        "    sw   x7,   8(x31)\n"
        "    sw   x8,  12(x31)\n"
        "    sw   x9,  16(x31)\n"
        "    sw   x10, 20(x31)\n"
        "    sw   x11, 24(x31)\n"
        "    sw   x12, 28(x31)\n"
        "    sw   x13, 32(x31)\n"
        "    sw   x14, 36(x31)\n"
        "    sw   x15, 40(x31)\n"
        "    sw   x16, 44(x31)\n"
        "    sw   x17, 48(x31)\n"
        "    sw   x18, 52(x31)\n"
        "    sw   x19, 56(x31)\n"
        "    sw   x20, 60(x31)\n"
        "    sw   x21, 64(x31)\n"
        "    sw   x22, 68(x31)\n"
        "    sw   x23, 72(x31)\n"
        "    sw   x24, 76(x31)\n"
        "    sw   x25, 80(x31)\n"
        "    sw   x26, 84(x31)\n"
        "    sw   x27, 88(x31)\n"
        "    sw   x28, 92(x31)\n"
        "    sw   x29, 96(x31)\n"
        "\n"
        "    # halt via CharDev EOT at 0x10000000\n"
        "    lui  x30, 0x10000\n"
        "    li   x31, 4\n"
        "    sb   x31, 0(x30)\n"
        ".Lhalt: j .Lhalt\n"
    )


# ── Docker execution ─────────────────────────────────────────────────────────

def run_in_docker(asm_source):
    """Compile and simulate in Docker. Returns (output, success)."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test.S"
        src.write_text(asm_source, encoding="utf-8")

        gcc_flags = [
            "-march=rv32im_zicsr", "-mabi=ilp32",
            "-ffreestanding", "-nostdlib",
            "-Wl,-Ttext=0x80000000",
        ]

        # Step 1: compile
        r = subprocess.run([
            "docker", "run", "--rm", "-v", f"{tmp}:/work", DOCKER_IMAGE,
            "riscv32-unknown-elf-gcc", *gcc_flags,
            "-o", "/work/test.elf", "/work/test.S",
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return r.stderr, False

        # Step 2: extract raw binary
        r = subprocess.run([
            "docker", "run", "--rm", "-v", f"{tmp}:/work", DOCKER_IMAGE,
            "riscv32-unknown-elf-objcopy", "-O", "binary",
            "/work/test.elf", "/work/test.bin",
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return r.stderr, False

        # Step 3: simulate on Proteus
        r = subprocess.run([
            "docker", "run", "--rm", "-v", f"{tmp}:/work", DOCKER_IMAGE,
            "/proteus/sim/build/sim", "/work/test.bin",
        ], capture_output=True, text=True, timeout=30)

        return r.stdout + r.stderr, (r.returncode == 0)


# ── Output parsing ───────────────────────────────────────────────────────────

def parse_memtrace(output):
    """Extract [(type, addr)] from the MEMTRACE block."""
    if "[MEMTRACE_START]" not in output or "[MEMTRACE_END]" not in output:
        return []
    body = output[
        output.index("[MEMTRACE_START]") + len("[MEMTRACE_START]"):
        output.index("[MEMTRACE_END]")
    ]
    trace = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("[MEMTRACE]"):
            continue
        parts = {}
        for token in line.split()[1:]:
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k] = v
        if "type" in parts and "addr" in parts:
            trace.append((parts["type"], int(parts["addr"], 16)))
    return trace


def print_output(output, ok):
    """Display MEMTRACE summary and REGDUMP from Proteus output."""
    print(f"\n{'[OK]' if ok else '[KO]'} Proteus output:")

    if not output.strip():
        print("  (no output)")
        return

    trace = parse_memtrace(output)
    if trace:
        print(f"\nMemory trace ({len(trace)} accesses):")
        for t, addr in trace[:10]:
            print(f"  {t:5s}  0x{addr:08x}")
        if len(trace) > 10:
            print(f"  ... ({len(trace)} total)")

    if "[REGDUMP]" in output and "[/REGDUMP]" in output:
        dump  = output[output.index("[REGDUMP]") + len("[REGDUMP]"):output.index("[/REGDUMP]")]
        after = output[output.index("[/REGDUMP]") + len("[/REGDUMP]"):].strip()
        print("\nRegister state after execution:")
        for line in dump.splitlines():
            line = line.strip()
            if "=" in line:
                reg, val = line.split(" = ")
                print(f"  {reg} = {val}")
        if after:
            print(f"\n{after}")
    else:
        print(output)
        print("\n[!] No REGDUMP found — rebuild Docker image with the updated main.cpp")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run one RV32IM instruction on Proteus via Docker")
    ap.add_argument("--isa",  default="riscv_isa_rv32.json", help="ISA spec file")
    ap.add_argument("--seed", type=int, default=None,         help="Random seed")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    with open(args.isa, encoding="utf-8") as f:
        instructions = json.load(f)

    # Skip control-flow instructions (branches need labels not present in this harness).
    testable = [i for i in instructions if not i.get("is_control_flow", False)]

    instr      = random.choice(testable)
    instr_line = format_instruction(instr)

    print(f"Instruction : {instr['name']}")
    print(f"Assembly    : {instr_line}")
    print(f"Category    : {instr['category']}")

    asm = build_asm_source(instr_line)
    print("\n--- Assembly source ---")
    print(asm)
    print("-----------------------\n")

    print("Running on Proteus via Docker...")
    output, ok = run_in_docker(asm)
    print_output(output, ok)


if __name__ == "__main__":
    main()