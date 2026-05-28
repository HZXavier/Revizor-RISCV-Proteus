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
    lo, hi = int(m.group(1)), int(m.group(2))
    return random.randint(lo, hi)


def pick_label() -> str:
    # Placeholder label so branch instructions have a valid target during assembly
    return "lbl_target"


def format_instruction(instr: Dict[str, Any]) -> str:
    """Build the assembly text for one instruction based on its category."""
    name = instr["name"]
    cat = instr["category"]
    operands = instr["operands"]

    # Loads: LX rd, offset(rs1)
    if cat == "RV64I-LOAD":
        rd     = pick_register(operands[0])
        offset = pick_immediate(operands[1])
        rs1    = pick_register(operands[2])
        return f"{name.lower()} {rd}, {offset}({rs1})"

    # Stores: SX rs2, offset(rs1)
    if cat == "RV64I-STORE":
        rs2    = pick_register(operands[0])
        offset = pick_immediate(operands[1])
        rs1    = pick_register(operands[2])
        return f"{name.lower()} {rs2}, {offset}({rs1})"

    # Conditional branches: BXX rs1, rs2, label
    if cat == "RV64I-COND-BR":
        rs1 = pick_register(operands[0])
        rs2 = pick_register(operands[1])
        lbl = pick_label()
        return f"{name.lower()} {rs1}, {rs2}, {lbl}"

    # JAL: jal rd, label
    if cat == "RV64I-UNCOND-BR":
        rd  = pick_register(operands[0])
        lbl = pick_label()
        return f"{name.lower()} {rd}, {lbl}"

    # JALR: jalr rd, imm(rs1)
    if cat == "RV64I-INDIRECT-BR":
        rd  = pick_register(operands[0])
        rs1 = pick_register(operands[1])
        imm = pick_immediate(operands[2])
        return f"{name.lower()} {rd}, {imm}({rs1})"

    # LUI / AUIPC: OP rd, imm20
    if cat == "RV64I-UPPER-IMM":
        rd  = pick_register(operands[0])
        imm = pick_immediate(operands[1])
        return f"{name.lower()} {rd}, {imm}"

    # Default: OP rd, rs1, rs2 or OP rd, rs1, imm (R-type, I-type, shifts, M-ext)
    parts = []
    for op in operands:
        if op["type_"] == "REG":
            parts.append(pick_register(op))
        elif op["type_"] == "IMM":
            parts.append(str(pick_immediate(op)))
        else:
            raise ValueError(f"Unexpected operand type: {op['type_']}")
    return f"{name.lower()} {', '.join(parts)}"


def build_asm_source(asm_line: str, needs_label: bool) -> str:
    """Wrap one instruction in a minimal .s file the assembler will accept."""
    label_block = ""
    if needs_label:
        # Branch targets need a real label with at least one instruction after them
        label_block = "lbl_target:\n    nop\n"

    return (
        ".section .text\n"
        ".globl _start\n"
        "_start:\n"
        f"    {asm_line}\n"
        f"{label_block}"
    )


def assemble(asm_source: str) -> Tuple[bool, str]:
    """Write source to a temp file and invoke riscv64-elf-as. Returns (success, message)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "test.s"
        obj_path = Path(tmpdir) / "test.o"
        src_path.write_text(asm_source, encoding="utf-8")

        try:
            result = subprocess.run(
                ["riscv64-elf-as",
                 "-march=rv64im",
                 "-o", str(obj_path),
                 str(src_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            return False, "riscv64-elf-as not found. Install binutils-riscv64-linux-gnu."
        except subprocess.TimeoutExpired:
            return False, "Assembler timed out after 10 seconds"

        if result.returncode == 0:
            return True, "OK"
        return False, result.stderr.strip() or result.stdout.strip()


def needs_label(instr: Dict[str, Any]) -> bool:
    return any(op["type_"] == "LABEL" for op in instr["operands"])


def run_one(instr: Dict[str, Any], verbose: bool = True) -> bool:
    """Generate and assemble one instruction. Returns True if the assembler accepts it."""
    asm_line = format_instruction(instr)
    source = build_asm_source(asm_line, needs_label(instr))
    ok, msg = assemble(source)

    if verbose:
        status = "[OK]" if ok else "[KO]"
        print(f"{status} {instr['name']:8s}  ->  {asm_line}")
        if not ok:
            print(f"     error: {msg}")

    return ok


def run_all(instructions: List[Dict[str, Any]]) -> None:
    """Test every instruction in the JSON one by one."""
    print(f"Testing {len(instructions)} instructions...\n")
    failures = []
    for instr in instructions:
        if not run_one(instr, verbose=True):
            failures.append(instr["name"])

    print("\n" + "=" * 60)
    if not failures:
        print(f"[OK] All {len(instructions)} instructions accepted.")
    else:
        print(f"[KO] {len(failures)} instruction(s) rejected:")
        for name in failures:
            print(f"  - {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and test RISC-V instructions")
    parser.add_argument("json_path", help="Path to the ISA JSON file")
    parser.add_argument("--instruction", "-i", help="Test a specific instruction by name")
    parser.add_argument("--all", "-a", action="store_true", help="Test all instructions in the JSON")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    with open(args.json_path, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    if args.all:
        run_all(instructions)
    elif args.instruction:
        matches = [i for i in instructions if i["name"] == args.instruction.upper()]
        if not matches:
            print(f"Instruction '{args.instruction}' not found in the JSON.")
            return 1
        run_one(matches[0])
    else:
        # Default: pick one random instruction
        instr = random.choice(instructions)
        run_one(instr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
