"""
Generates riscv_isa.json: a structured spec for RV64I + RV64M instructions.
Used by filter_rv32.py to produce the RV32IM spec for test generation.

Excluded extensions: A (atomic), F/D (float), C (compressed), Zicsr, Zifencei.
"""

import json
from typing import List, Dict, Any

# x0 is hardwired to zero; x1-x4 are ABI-reserved (ra, sp, gp, tp).
DEST_REGS = [f"x{i}" for i in range(5, 32)]
SRC_REGS  = [f"x{i}" for i in range(0, 32)]

XLEN = 64


def reg_operand(name: str, src: bool, dest: bool,
                values: List[str] = None) -> Dict[str, Any]:
    return {
        "name":   name,
        "type_":  "REG",
        "src":    src,
        "dest":   dest,
        "width":  XLEN,
        "values": values if values is not None else (DEST_REGS if dest else SRC_REGS),
    }


def imm_operand(name: str, width: int, lo: int, hi: int,
                is_signed: bool = True) -> Dict[str, Any]:
    return {
        "name":      name,
        "type_":     "IMM",
        "src":       True,
        "dest":      False,
        "width":     width,
        "is_signed": is_signed,
        "values":    [f"[{lo}-{hi}]"],
    }


def label_operand(name: str = "label") -> Dict[str, Any]:
    return {
        "name":   name,
        "type_":  "LABEL",
        "src":    True,
        "dest":   False,
        "width":  0,
        "values": [],
    }


def make_instruction(name: str, category: str, is_control_flow: bool,
                     operands: List[Dict[str, Any]],
                     implicit_operands: List[Dict[str, Any]] = None,
                     comment: str = "") -> Dict[str, Any]:
    instr = {
        "name":             name,
        "category":         category,
        "is_control_flow":  is_control_flow,
        "operands":         operands,
        "implicit_operands": implicit_operands or [],
    }
    if comment:
        instr["_comment"] = comment
    return instr


isa_spec: List[Dict[str, Any]] = []


# ── RV64I: base integer ───────────────────────────────────────────────────────

# R-type: rd = rs1 OP rs2
for mnem in ["ADD", "SUB", "AND", "OR", "XOR", "SLL", "SRL", "SRA", "SLT", "SLTU"]:
    isa_spec.append(make_instruction(
        name=mnem, category="RV64I-ARITH", is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            reg_operand("rs2", src=True,  dest=False),
        ],
    ))

# *W variants: operate on 32 bits, result sign-extended to 64 bits
for mnem in ["ADDW", "SUBW", "SLLW", "SRLW", "SRAW"]:
    isa_spec.append(make_instruction(
        name=mnem, category="RV64I-ARITH-W", is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            reg_operand("rs2", src=True,  dest=False),
        ],
    ))

# I-type arithmetic: rd = rs1 OP imm12
for mnem in ["ADDI", "ANDI", "ORI", "XORI", "SLTI", "SLTIU"]:
    isa_spec.append(make_instruction(
        name=mnem, category="RV64I-ARITH-IMM", is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            imm_operand("imm12", width=12, lo=-2048, hi=2047),
        ],
    ))

# ADDIW: 32-bit add, result sign-extended to 64 bits
isa_spec.append(make_instruction(
    name="ADDIW", category="RV64I-ARITH-IMM-W", is_control_flow=False,
    operands=[
        reg_operand("rd",  src=False, dest=True),
        reg_operand("rs1", src=True,  dest=False),
        imm_operand("imm12", width=12, lo=-2048, hi=2047),
    ],
))

# 64-bit shifts: shamt is 6 bits (0-63)
for mnem in ["SLLI", "SRLI", "SRAI"]:
    isa_spec.append(make_instruction(
        name=mnem, category="RV64I-SHIFT-IMM", is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            imm_operand("shamt", width=6, lo=0, hi=63, is_signed=False),
        ],
    ))

# 32-bit shifts: shamt is 5 bits (0-31), result sign-extended
for mnem in ["SLLIW", "SRLIW", "SRAIW"]:
    isa_spec.append(make_instruction(
        name=mnem, category="RV64I-SHIFT-IMM-W", is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            imm_operand("shamt", width=5, lo=0, hi=31, is_signed=False),
        ],
    ))

# LUI: rd = imm20 << 12
isa_spec.append(make_instruction(
    name="LUI", category="RV64I-UPPER-IMM", is_control_flow=False,
    operands=[
        reg_operand("rd", src=False, dest=True),
        imm_operand("imm20", width=20, lo=0, hi=1048575, is_signed=False),
    ],
))

# AUIPC: rd = PC + (imm20 << 12)
isa_spec.append(make_instruction(
    name="AUIPC", category="RV64I-UPPER-IMM", is_control_flow=False,
    operands=[
        reg_operand("rd", src=False, dest=True),
        imm_operand("imm20", width=20, lo=0, hi=1048575, is_signed=False),
    ],
    implicit_operands=[
        {"name": "PC", "type_": "REG", "src": True, "dest": False,
         "width": XLEN, "values": ["PC"]},
    ],
))

# Loads: rd = MEM[rs1 + offset]
LOADS = [
    ("LB",  8),   # byte, sign-extended
    ("LBU", 8),   # byte, zero-extended
    ("LH",  16),
    ("LHU", 16),
    ("LW",  32),
    ("LWU", 32),  # RV64 only
    ("LD",  64),  # RV64 only
]
for mnem, bits in LOADS:
    isa_spec.append(make_instruction(
        name=mnem, category="RV64I-LOAD", is_control_flow=False,
        operands=[
            reg_operand("rd",     src=False, dest=True),
            imm_operand("offset", width=12, lo=-2048, hi=2047),
            reg_operand("rs1",    src=True,  dest=False),
        ],
        comment=f"Load {bits}-bit value from MEM[rs1+offset]",
    ))

# Stores: MEM[rs1 + offset] = rs2
STORES = [("SB", 8), ("SH", 16), ("SW", 32), ("SD", 64)]
for mnem, bits in STORES:
    isa_spec.append(make_instruction(
        name=mnem, category="RV64I-STORE", is_control_flow=False,
        operands=[
            reg_operand("rs2",    src=True, dest=False),
            imm_operand("offset", width=12, lo=-2048, hi=2047),
            reg_operand("rs1",    src=True, dest=False),
        ],
        comment=f"Store {bits}-bit value to MEM[rs1+offset]",
    ))

# Conditional branches
PC_IMPLICIT = [{"name": "PC", "type_": "REG", "src": True, "dest": False,
                "width": XLEN, "values": ["PC"]}]

for mnem in ["BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"]:
    isa_spec.append(make_instruction(
        name=mnem, category="RV64I-COND-BR", is_control_flow=True,
        operands=[
            reg_operand("rs1", src=True, dest=False),
            reg_operand("rs2", src=True, dest=False),
            label_operand("label"),
        ],
        implicit_operands=PC_IMPLICIT,
    ))

# JAL: rd = PC+4; PC = PC + offset
isa_spec.append(make_instruction(
    name="JAL", category="RV64I-UNCOND-BR", is_control_flow=True,
    operands=[
        reg_operand("rd", src=False, dest=True),
        label_operand("label"),
    ],
    implicit_operands=PC_IMPLICIT,
))

# JALR: rd = PC+4; PC = (rs1 + imm12) & ~1
isa_spec.append(make_instruction(
    name="JALR", category="RV64I-INDIRECT-BR", is_control_flow=True,
    operands=[
        reg_operand("rd",  src=False, dest=True),
        reg_operand("rs1", src=True,  dest=False),
        imm_operand("imm12", width=12, lo=-2048, hi=2047),
    ],
    implicit_operands=PC_IMPLICIT,
))


# ── RV64M: multiply / divide ──────────────────────────────────────────────────

for mnem in ["MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU"]:
    isa_spec.append(make_instruction(
        name=mnem, category="RV64M-MULDIV", is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            reg_operand("rs2", src=True,  dest=False),
        ],
    ))

# *W variants: 32-bit operation, result sign-extended to 64 bits
for mnem in ["MULW", "DIVW", "DIVUW", "REMW", "REMUW"]:
    isa_spec.append(make_instruction(
        name=mnem, category="RV64M-MULDIV-W", is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            reg_operand("rs2", src=True,  dest=False),
        ],
    ))


# ── Output ───────────────────────────────────────────────────────────────────

def main(output_path: str = "riscv_isa.json") -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(isa_spec, f, indent=2)

    by_cat: Dict[str, int] = {}
    for instr in isa_spec:
        by_cat[instr["category"]] = by_cat.get(instr["category"], 0) + 1

    print(f"[OK] {len(isa_spec)} instructions written to {output_path}")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:30s} {n:3d}")


if __name__ == "__main__":
    main()