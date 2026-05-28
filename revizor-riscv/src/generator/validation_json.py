import json
import re
import sys
from typing import Any, Dict, List, Tuple


VALID_OPERAND_TYPES = {"REG", "MEM", "IMM", "LABEL", "AGEN", "FLAGS", "COND"}

REQUIRED_INSTR_FIELDS = {"name", "category", "is_control_flow",
                         "operands", "implicit_operands"}
REQUIRED_OPERAND_FIELDS = {"type_", "src", "dest", "width"}

# Matches "[lo-hi]" where lo may be negative
RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")


def check_operand(op: Dict[str, Any], context: str,
                  errors: List[str], warnings: List[str]) -> None:
    missing = REQUIRED_OPERAND_FIELDS - set(op.keys())
    if missing:
        errors.append(f"{context}: missing fields {missing}")
        return

    if op["type_"] not in VALID_OPERAND_TYPES:
        errors.append(f"{context}: unknown type_ '{op['type_']}'")
        return

    if not isinstance(op["src"], bool) or not isinstance(op["dest"], bool):
        errors.append(f"{context}: src and dest must be booleans")

    # Every operand should be a source or a destination (or both)
    if not op["src"] and not op["dest"] and op["type_"] not in {"COND", "FLAGS"}:
        warnings.append(f"{context}: operand is neither src nor dest")

    width = op["width"]
    if not isinstance(width, int) or width < 0:
        errors.append(f"{context}: width must be an integer >= 0")

    if op["type_"] == "REG":
        if width not in {0, 32, 64}:
            warnings.append(f"{context}: unusual width={width} for a REG")
        values = op.get("values", [])
        if not values:
            errors.append(f"{context}: REG has no allowed values")

    elif op["type_"] == "IMM":
        check_immediate(op, context, errors, warnings)

    elif op["type_"] == "LABEL":
        if op.get("dest", False):
            errors.append(f"{context}: LABEL cannot be a destination")


def check_immediate(op: Dict[str, Any], context: str,
                    errors: List[str], warnings: List[str]) -> None:
    width = op["width"]
    is_signed = op.get("is_signed", True)
    values = op.get("values", [])

    if not values:
        errors.append(f"{context}: IMM has no value range")
        return

    for val in values:
        m = RANGE_RE.match(val)
        if not m:
            errors.append(f"{context}: malformed range '{val}'")
            continue
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            errors.append(f"{context}: inverted range [{lo}, {hi}]")

        # Check that the range fits within the declared bit width
        if is_signed:
            min_allowed = -(1 << (width - 1))
            max_allowed = (1 << (width - 1)) - 1
        else:
            min_allowed = 0
            max_allowed = (1 << width) - 1

        if lo < min_allowed or hi > max_allowed:
            errors.append(
                f"{context}: range [{lo}, {hi}] out of bounds for "
                f"{'signed' if is_signed else 'unsigned'} {width}-bit IMM "
                f"([{min_allowed}, {max_allowed}])"
            )


def check_instruction(instr: Dict[str, Any], idx: int,
                      errors: List[str], warnings: List[str]) -> None:
    name = instr.get("name", f"<unnamed #{idx}>")

    missing = REQUIRED_INSTR_FIELDS - set(instr.keys())
    if missing:
        errors.append(f"{name}: missing fields {missing}")
        return

    if not isinstance(instr["operands"], list):
        errors.append(f"{name}: 'operands' must be a list")
        return
    if not isinstance(instr["implicit_operands"], list):
        errors.append(f"{name}: 'implicit_operands' must be a list")
        return
    if not isinstance(instr["is_control_flow"], bool):
        errors.append(f"{name}: 'is_control_flow' must be a boolean")

    for i, op in enumerate(instr["operands"]):
        check_operand(op, f"{name}.operands[{i}]", errors, warnings)
    for i, op in enumerate(instr["implicit_operands"]):
        check_operand(op, f"{name}.implicit_operands[{i}]", errors, warnings)

    # A control-flow instruction must have a LABEL operand or be an indirect branch
    if instr["is_control_flow"]:
        has_label = any(o["type_"] == "LABEL" for o in instr["operands"])
        is_indirect = "INDIRECT" in instr["category"].upper()
        if not has_label and not is_indirect:
            warnings.append(
                f"{name}: marked as control_flow but has no LABEL and is not indirect"
            )


def check_coverage(instructions: List[Dict[str, Any]],
                   warnings: List[str]) -> None:
    """Check that all major instruction families are present."""
    names = {i["name"] for i in instructions}
    cats = {i["category"] for i in instructions}

    if not any(c.endswith("ARITH") or c.endswith("ARITH-IMM") for c in cats):
        warnings.append("No arithmetic instructions found")
    if not any("LOAD" in c for c in cats):
        warnings.append("No load instructions found")
    if not any("STORE" in c for c in cats):
        warnings.append("No store instructions found")
    if not any("BR" in c for c in cats):
        warnings.append("No branch instructions found")

    expected = {"ADD", "ADDI", "LD", "SD", "BEQ", "JAL"}
    missing = expected - names
    if missing:
        warnings.append(f"Missing core RV64I instructions: {missing}")


def check_duplicates(instructions: List[Dict[str, Any]],
                     warnings: List[str]) -> None:
    seen = {}
    for i, instr in enumerate(instructions):
        name = instr["name"]
        if name in seen:
            warnings.append(f"Duplicate: '{name}' at indices {seen[name]} and {i}")
        else:
            seen[name] = i


def validate(path: str) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        errors.append(f"File not found: {path}")
        return errors, warnings
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return errors, warnings

    if not isinstance(data, list):
        errors.append("Root must be a list of instructions")
        return errors, warnings

    if not data:
        errors.append("File contains no instructions")
        return errors, warnings

    for idx, instr in enumerate(data):
        if not isinstance(instr, dict):
            errors.append(f"Element {idx}: not an object")
            continue
        check_instruction(instr, idx, errors, warnings)

    check_duplicates(data, warnings)
    check_coverage(data, warnings)

    return errors, warnings


def main(path: str) -> int:
    print(f"Validating: {path}")
    print("-" * 60)

    errors, warnings = validate(path)

    if errors:
        print(f"\n[X] {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[OK] No structural errors.")

    if warnings:
        print(f"\n[!] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("[OK] No warnings.")

    print("-" * 60)
    if errors:
        print("Result: INVALID")
        return 1
    print("Result: VALID" + (" (with warnings)" if warnings else ""))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 validate_isa.py <file.json>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
