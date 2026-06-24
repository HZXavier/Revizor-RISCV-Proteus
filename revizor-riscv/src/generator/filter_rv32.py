"""
Filters riscv_isa.json to keep only RV32IM-compatible instructions.

"""

import argparse
import json
import sys


RV64_ONLY_CATEGORIES = {
    "RV64I-ARITH-W",
    "RV64I-ARITH-IMM-W",
    "RV64I-SHIFT-IMM-W",
    "RV64M-MULDIV-W",
}

RV64_ONLY_NAMES = {"LD", "SD", "LWU"}


def filter_to_rv32(instructions):
    """Drop RV64-only entries and adjust the remaining ones for RV32."""
    rv32 = [
        i for i in instructions
        if i["category"] not in RV64_ONLY_CATEGORIES
        and i["name"] not in RV64_ONLY_NAMES
    ]

    for instr in rv32:
        # Fix shamt width: 5 bits in RV32 vs 6 bits in RV64.
        if instr["category"] == "RV64I-SHIFT-IMM":
            for op in instr["operands"]:
                if op["name"] == "shamt":
                    op["width"]  = 5
                    op["values"] = ["[0-31]"]

        # Rename category prefix.
        instr["category"] = instr["category"].replace("RV64", "RV32")

    return rv32


def main():
    ap = argparse.ArgumentParser(description="Filter RV64IM spec down to RV32IM")
    ap.add_argument("--input",  default="riscv_isa.json",      help="RV64IM spec to read")
    ap.add_argument("--output", default="riscv_isa_rv32.json", help="RV32IM spec to write")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    rv32 = filter_to_rv32(instructions)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rv32, f, indent=2)

    print(f"[OK] {len(rv32)}/{len(instructions)} instructions kept (RV32IM) -> {args.output}")
    by_cat = {}
    for i in rv32:
        by_cat[i["category"]] = by_cat.get(i["category"], 0) + 1
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:30s}  {n}")


if __name__ == "__main__":
    sys.exit(main() or 0)