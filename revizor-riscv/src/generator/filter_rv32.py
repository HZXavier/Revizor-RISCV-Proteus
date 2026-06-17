"""
Filters riscv_isa.json to keep only RV32IM-compatible instructions.
Removes: *W variants, LD, SD, LWU (RV64-only).
Output: riscv_isa_rv32.json
"""
import json

RV64_ONLY_CATEGORIES = {
    "RV64I-ARITH-W",
    "RV64I-ARITH-IMM-W",
    "RV64I-SHIFT-IMM-W",
    "RV64M-MULDIV-W",
}
RV64_ONLY_NAMES = {"LD", "SD", "LWU"}

with open("riscv_isa.json") as f:
    instructions = json.load(f)

rv32 = [
    i for i in instructions
    if i["category"] not in RV64_ONLY_CATEGORIES
    and i["name"] not in RV64_ONLY_NAMES
]

# Also fix shift shamt width: RV32 shifts use 5-bit shamt, not 6-bit
for instr in rv32:
    if instr["category"] == "RV64I-SHIFT-IMM":
        for op in instr["operands"]:
            if op["name"] == "shamt":
                op["width"] = 5
                op["values"] = ["[0-31]"]

# Rename categories from RV64I to RV32I for clarity
for instr in rv32:
    instr["category"] = instr["category"].replace("RV64", "RV32")

with open("riscv_isa_rv32.json", "w") as f:
    json.dump(rv32, f, indent=2)

print(f"[OK] {len(rv32)}/{len(instructions)} instructions kept (RV32IM)")
by_cat = {}
for i in rv32:
    by_cat[i["category"]] = by_cat.get(i["category"], 0) + 1
for cat, n in sorted(by_cat.items()):
    print(f"  {cat:30s}  {n}")