"""
Generates a JSON spec for RISC-V RV64I + RV64M instructions.
Output: riscv_isa.json, used to randomly generate and assemble RISC-V programs.
Covers: RV64I (base integer) and RV64M (multiply/divide).
Not covered: A (atomic), F/D (float), C (compressed), Zicsr, Zifencei.
"""

import json
from typing import List, Dict, Any

# x0 is always zero (read-only, useless as destination)
# x1-x4 are reserved (ra, sp, gp, tp)
# x5-x31 are free to use
DEST_REGS = [f"x{i}" for i in range(5, 32)]
SRC_REGS  = [f"x{i}" for i in range(0, 32)]

XLEN = 64  # register width in RV64


def reg_operand(name: str, src: bool, dest: bool,
                values: List[str] = None) -> Dict[str, Any]:
    return {
        "name": name,
        "type_": "REG",
        "src": src,
        "dest": dest,
        "width": XLEN,
        "values": values if values is not None else (DEST_REGS if dest else SRC_REGS),
    }


def imm_operand(name: str, width: int, lo: int, hi: int,
                is_signed: bool = True) -> Dict[str, Any]:
    # lo/hi are inclusive bounds
    return {
        "name": name,
        "type_": "IMM",
        "src": True,
        "dest": False,
        "width": width,
        "is_signed": is_signed,
        "values": [f"[{lo}-{hi}]"],
    }


def label_operand(name: str = "label") -> Dict[str, Any]:
    return {
        "name": name,
        "type_": "LABEL",
        "src": True,
        "dest": False,
        "width": 0,
        "values": [],
    }


def make_instruction(name: str, category: str, is_control_flow: bool,
                     operands: List[Dict[str, Any]],
                     implicit_operands: List[Dict[str, Any]] = None,
                     comment: str = "") -> Dict[str, Any]:
    instr = {
        "name": name,
        "category": category,
        "is_control_flow": is_control_flow,
        "operands": operands,
        "implicit_operands": implicit_operands if implicit_operands else [],
    }
    if comment:
        instr["_comment"] = comment
    return instr


isa_spec: List[Dict[str, Any]] = []


# R-type: OP rd, rs1, rs2
R_TYPE = ["ADD", "SUB", "AND", "OR", "XOR", "SLL", "SRL", "SRA", "SLT", "SLTU"]
for mnem in R_TYPE:
    isa_spec.append(make_instruction(
        name=mnem,
        category="RV64I-ARITH",
        is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            reg_operand("rs2", src=True,  dest=False),
        ],
    ))

# 32-bit R-type: result is sign-extended to 64 bits
R_TYPE_W = ["ADDW", "SUBW", "SLLW", "SRLW", "SRAW"]
for mnem in R_TYPE_W:
    isa_spec.append(make_instruction(
        name=mnem,
        category="RV64I-ARITH-W",
        is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            reg_operand("rs2", src=True,  dest=False),
        ],
        comment="Operates on 32 bits, result sign-extended to 64 bits",
    ))


# I-type arithmetic: OP rd, rs1, imm12
I_TYPE_ARITH = ["ADDI", "ANDI", "ORI", "XORI", "SLTI", "SLTIU"]
for mnem in I_TYPE_ARITH:
    isa_spec.append(make_instruction(
        name=mnem,
        category="RV64I-ARITH-IMM",
        is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            imm_operand("imm12", width=12, lo=-2048, hi=2047, is_signed=True),
        ],
    ))

# 32-bit immediate add, result sign-extended to 64 bits
isa_spec.append(make_instruction(
    name="ADDIW",
    category="RV64I-ARITH-IMM-W",
    is_control_flow=False,
    operands=[
        reg_operand("rd",  src=False, dest=True),
        reg_operand("rs1", src=True,  dest=False),
        imm_operand("imm12", width=12, lo=-2048, hi=2047, is_signed=True),
    ],
    comment="32-bit add with sign-extension to 64 bits",
))


# 64-bit immediate shifts, shamt is 6 bits
I_TYPE_SHIFT = ["SLLI", "SRLI", "SRAI"]
for mnem in I_TYPE_SHIFT:
    isa_spec.append(make_instruction(
        name=mnem,
        category="RV64I-SHIFT-IMM",
        is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            imm_operand("shamt", width=6, lo=0, hi=63, is_signed=False),
        ],
        comment="64-bit shift, shamt is 6 bits",
    ))

# 32-bit immediate shifts, shamt is 5 bits
I_TYPE_SHIFT_W = ["SLLIW", "SRLIW", "SRAIW"]
for mnem in I_TYPE_SHIFT_W:
    isa_spec.append(make_instruction(
        name=mnem,
        category="RV64I-SHIFT-IMM-W",
        is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            imm_operand("shamt", width=5, lo=0, hi=31, is_signed=False),
        ],
        comment="32-bit shift, shamt is 5 bits, result sign-extended",
    ))


# LUI: load upper 20-bit immediate into rd (shifted left by 12)
isa_spec.append(make_instruction(
    name="LUI",
    category="RV64I-UPPER-IMM",
    is_control_flow=False,
    operands=[
        reg_operand("rd", src=False, dest=True),
        imm_operand("imm20", width=20, lo=0, hi=1048575, is_signed=False),
    ],
))

# AUIPC: add upper immediate to PC
isa_spec.append(make_instruction(
    name="AUIPC",
    category="RV64I-UPPER-IMM",
    is_control_flow=False,
    operands=[
        reg_operand("rd", src=False, dest=True),
        imm_operand("imm20", width=20, lo=0, hi=1048575, is_signed=False),
    ],
    implicit_operands=[
        {"name": "PC", "type_": "REG", "src": True, "dest": False,
         "width": XLEN, "values": ["PC"]}
    ],
))


# Loads: LX rd, offset(rs1) — reads from memory at address rs1+offset
LOADS = [
    ("LB",  8,  True),   # byte, signed
    ("LBU", 8,  False),  # byte, unsigned
    ("LH",  16, True),
    ("LHU", 16, False),
    ("LW",  32, True),
    ("LWU", 32, False),  # RV64 only
    ("LD",  64, True),   # RV64 only
]
for mnem, mem_bits, _signed in LOADS:
    isa_spec.append(make_instruction(
        name=mnem,
        category="RV64I-LOAD",
        is_control_flow=False,
        operands=[
            reg_operand("rd",     src=False, dest=True),
            imm_operand("offset", width=12, lo=-2048, hi=2047, is_signed=True),
            reg_operand("rs1",    src=True,  dest=False),
        ],
        comment=f"Load {mem_bits} bits from MEM[rs1+offset]",
    ))


# Stores: SX rs2, offset(rs1) — writes to memory at address rs1+offset
STORES = [
    ("SB", 8),
    ("SH", 16),
    ("SW", 32),
    ("SD", 64),  # RV64 only
]
for mnem, mem_bits in STORES:
    isa_spec.append(make_instruction(
        name=mnem,
        category="RV64I-STORE",
        is_control_flow=False,
        operands=[
            reg_operand("rs2",    src=True, dest=False),
            imm_operand("offset", width=12, lo=-2048, hi=2047, is_signed=True),
            reg_operand("rs1",    src=True, dest=False),
        ],
        comment=f"Store {mem_bits} bits to MEM[rs1+offset]",
    ))


# Conditional branches: BXX rs1, rs2, label
BRANCHES = ["BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"]
for mnem in BRANCHES:
    isa_spec.append(make_instruction(
        name=mnem,
        category="RV64I-COND-BR",
        is_control_flow=True,
        operands=[
            reg_operand("rs1", src=True, dest=False),
            reg_operand("rs2", src=True, dest=False),
            label_operand("label"),
        ],
        implicit_operands=[
            {"name": "PC", "type_": "REG", "src": True, "dest": False,
             "width": XLEN, "values": ["PC"]}
        ],
    ))


# JAL: unconditional jump, saves PC+4 in rd
isa_spec.append(make_instruction(
    name="JAL",
    category="RV64I-UNCOND-BR",
    is_control_flow=True,
    operands=[
        reg_operand("rd", src=False, dest=True),
        label_operand("label"),
    ],
    implicit_operands=[
        {"name": "PC", "type_": "REG", "src": True, "dest": False,
         "width": XLEN, "values": ["PC"]}
    ],
))

# JALR: indirect jump to rs1+imm12, saves PC+4 in rd
isa_spec.append(make_instruction(
    name="JALR",
    category="RV64I-INDIRECT-BR",
    is_control_flow=True,
    operands=[
        reg_operand("rd",  src=False, dest=True),
        reg_operand("rs1", src=True,  dest=False),
        imm_operand("imm12", width=12, lo=-2048, hi=2047, is_signed=True),
    ],
    implicit_operands=[
        {"name": "PC", "type_": "REG", "src": True, "dest": False,
         "width": XLEN, "values": ["PC"]}
    ],
))


# RV64M: 64-bit multiply/divide
M_TYPE_64 = ["MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU"]
for mnem in M_TYPE_64:
    isa_spec.append(make_instruction(
        name=mnem,
        category="RV64M-MULDIV",
        is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            reg_operand("rs2", src=True,  dest=False),
        ],
        comment="64-bit multiply/divide",
    ))

# RV64M: 32-bit multiply/divide, result sign-extended to 64 bits
M_TYPE_32 = ["MULW", "DIVW", "DIVUW", "REMW", "REMUW"]
for mnem in M_TYPE_32:
    isa_spec.append(make_instruction(
        name=mnem,
        category="RV64M-MULDIV-W",
        is_control_flow=False,
        operands=[
            reg_operand("rd",  src=False, dest=True),
            reg_operand("rs1", src=True,  dest=False),
            reg_operand("rs2", src=True,  dest=False),
        ],
        comment="32-bit multiply/divide, result sign-extended to 64 bits",
    ))


def main(output_path: str = "riscv_isa.json") -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(isa_spec, f, indent=2)

    print(f"[OK] {len(isa_spec)} instructions written to {output_path}")
    by_cat: Dict[str, int] = {}
    for instr in isa_spec:
        by_cat[instr["category"]] = by_cat.get(instr["category"], 0) + 1
    print("Breakdown by category:")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:30s} {n:3d}")


if __name__ == "__main__":
    main()
