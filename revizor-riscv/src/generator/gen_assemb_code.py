"""
Validates riscv_isa.json by feeding each instruction to riscv64-elf-as.

Usage:
    python3 test_instructions.py riscv_isa.json --all
    python3 test_instructions.py riscv_isa.json -i ADD
    python3 test_instructions.py riscv_isa.json --seed 42
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")


def pick_register(operand: Dict[str, Any]) -> str:
    return random.choice(operand["values"])


def pick_immediate(operand: Dict[str, Any]) -> int:
    raw = operand["values"][0]
    m = RANGE_RE.match(raw)
    if not m:
        raise ValueError(f"Malformed range: {raw}")
    return random.randint(int(m.group(1)), int(m.group(2)))


def pick_label() -> str:
    # Placeholder target placed right after the test instruction.
    return "lbl_target"


def format_instruction(instr: Dict[str, Any]) -> str:
    """Render one instruction as an assembly line based on its category."""
    name = instr["name"].lower()
    cat  = instr["category"]
    ops  = instr["operands"]

    if cat == "RV64I-LOAD":
        # LX rd, offset(rs1)
        return f"{name} {pick_register(ops[0])}, {pick_immediate(ops[1])}({pick_register(ops[2])})"

    if cat == "RV64I-STORE":
        # SX rs2, offset(rs1)
        return f"{name} {pick_register(ops[0])}, {pick_immediate(ops[1])}({pick_register(ops[2])})"

    if cat == "RV64I-COND-BR":
        # BXX rs1, rs2, label
        return f"{name} {pick_register(ops[0])}, {pick_register(ops[1])}, {pick_label()}"

    if cat == "RV64I-UNCOND-BR":
        # JAL rd, label
        return f"{name} {pick_register(ops[0])}, {pick_label()}"

    if cat == "RV64I-INDIRECT-BR":
        # JALR rd, imm(rs1)
        return f"{name} {pick_register(ops[0])}, {pick_immediate(ops[2])}({pick_register(ops[1])})"

    if cat == "RV64I-UPPER-IMM":
        # LUI / AUIPC: OP rd, imm20
        return f"{name} {pick_register(ops[0])}, {pick_immediate(ops[1])}"

    # Default: positional REG/IMM operands (covers all ARITH, SHIFT, MULDIV variants).
    parts: List[str] = []
    for op in ops:
        if op["type_"] == "REG":
            parts.append(pick_register(op))
        elif op["type_"] == "IMM":
            parts.append(str(pick_immediate(op)))
        else:
            raise ValueError(f"Unexpected operand type: {op['type_']}")
    return f"{name} {', '.join(parts)}"


def needs_label(instr: Dict[str, Any]) -> bool:
    return any(op["type_"] == "LABEL" for op in instr["operands"])


def build_asm_source(asm_line: str, with_label: bool) -> str:
    """Wrap one instruction in a minimal .S file the assembler will accept."""
    label_block = "lbl_target:\n    nop\n" if with_label else ""
    return (
        ".section .text\n"
        ".globl _start\n"
        "_start:\n"
        f"    {asm_line}\n"
        f"{label_block}"
    )


def assemble(asm_source: str) -> Tuple[bool, str]:
    """Run riscv64-elf-as on the source. Returns (success, message)."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test.s"
        obj = Path(tmp) / "test.o"
        src.write_text(asm_source, encoding="utf-8")

        try:
            r = subprocess.run(
                ["riscv64-elf-as", "-march=rv64im", "-o", str(obj), str(src)],
                capture_output=True, text=True, timeout=10,
            )
        except FileNotFoundError:
            return False, "riscv64-elf-as not found. Install binutils for RISC-V."
        except subprocess.TimeoutExpired:
            return False, "Assembler timed out after 10s."

        if r.returncode == 0:
            return True, "OK"
        return False, (r.stderr or r.stdout).strip()


def run_one(instr: Dict[str, Any], verbose: bool = True) -> bool:
    """Generate one instruction and assemble it. Returns True if accepted."""
    line = format_instruction(instr)
    src  = build_asm_source(line, needs_label(instr))
    ok, msg = assemble(src)

    if verbose:
        status = "[OK]" if ok else "[KO]"
        print(f"{status} {instr['name']:8s}  ->  {line}")
        if not ok:
            print(f"     {msg}")

    return ok


def run_all(instructions: List[Dict[str, Any]]) -> None:
    """Test every instruction in the spec."""
    print(f"Testing {len(instructions)} instructions...\n")
    failures = [i["name"] for i in instructions if not run_one(i)]

    print("\n" + "=" * 60)
    if not failures:
        print(f"[OK] All {len(instructions)} instructions accepted.")
    else:
        print(f"[KO] {len(failures)} instruction(s) rejected:")
        for name in failures:
            print(f"  - {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate riscv_isa.json against the assembler")
    ap.add_argument("json_path", help="Path to the ISA JSON file")
    ap.add_argument("-i", "--instruction", help="Test a specific instruction by name")
    ap.add_argument("-a", "--all", action="store_true", help="Test every instruction")
    ap.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    with open(args.json_path, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    if args.all:
        run_all(instructions)
    elif args.instruction:
        matches = [i for i in instructions if i["name"] == args.instruction.upper()]
        if not matches:
            print(f"Instruction '{args.instruction}' not found.")
            return 1
        run_one(matches[0])
    else:
        run_one(random.choice(instructions))

    return 0


if __name__ == "__main__":
    sys.exit(main())