"""
CT-COND contract checker: detects Spectre variant 1 (PHT) patterns.

Based on Guarnieri et al. 2021 and Garcia 2023 (TFG section 3.3).
"""

import argparse
import random
import re
import subprocess
import tempfile
from pathlib import Path

from unicorn import Uc, UC_ARCH_RISCV, UC_MODE_RISCV32, UcError
from unicorn import UC_HOOK_CODE, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE
from unicorn.riscv_const import *

RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")

DOCKER_IMAGE = "proteus"
RAM_BASE     = 0x80000000
SANDBOX_ADDR = 0x80001000
RAM_SIZE     = 0x00100000   # 1 MiB

RESERVED_REGS = {"x30", "x31"}
TRACKED       = list(range(5, 30))
UC_REG        = {i: globals()[f"UC_RISCV_REG_X{i}"] for i in range(32)}


# ── Gadget generation ────────────────────────────────────────────────────────

def rand_reg(exclude=None):
    """Pick a random register from x5..x29, excluding reserved and given regs."""
    excl = (exclude or set()) | RESERVED_REGS
    return random.choice([f"x{i}" for i in TRACKED if f"x{i}" not in excl])


def generate_gadget():
    """
    Generate a random 3-instruction Spectre-v1 gadget:
      instr1 : arithmetic producing a secret-dependent value in rd1
      branch : conditional branch on rd1
      instr2 : load whose address depends on rd1 (transient)
    """
    rd1  = rand_reg()
    rs_a = rand_reg(exclude={rd1})
    rs_b = rand_reg(exclude={rd1, rs_a})

    op1    = random.choice(["add", "sub", "and", "or", "xor", "mul"])
    instr1 = f"{op1} {rd1}, {rs_a}, {rs_b}"

    # Use rs_b (random value) as the second operand for ordered comparisons,
    # not x0 (always zero), to generate more varied branch outcomes.
    branch_map = {
        "beq":  f"beq  {rd1}, x0, .skip",
        "bne":  f"bne  {rd1}, x0, .skip",
        "blt":  f"blt  {rd1}, {rs_b}, .skip",
        "bge":  f"bge  {rd1}, {rs_b}, .skip",
        "bltu": f"bltu {rd1}, {rs_b}, .skip",
        "bgeu": f"bgeu {rd1}, {rs_b}, .skip",
    }
    branch_name  = random.choice(list(branch_map.keys()))
    branch_instr = branch_map[branch_name]

    rd2    = rand_reg(exclude={rd1, rs_a, rs_b})
    op2    = random.choice(["lw", "lh", "lb", "lbu", "lhu"])
    instr2 = f"{op2} {rd2}, 0({rd1})"

    asm = (
        ".section .text\n"
        ".globl _start\n"
        "_start:\n"
        "    lui   x30, 0x80001\n"
        "\n"
        f"    {instr1}\n"
        "\n"
        f"    {branch_instr}\n"
        "\n"
        f"    # transient load: address depends on rd1 (secret-dependent)\n"
        f"    {instr2}\n"
        "\n"
        ".skip:\n"
        "    nop\n"
    )
    meta = {
        "instr1": instr1,
        "branch": branch_instr.strip(),
        "instr2": instr2,
        "rd1": rd1, "rs_a": rs_a, "rs_b": rs_b,
    }
    return asm, meta


# ── Docker compilation ───────────────────────────────────────────────────────

def compile_gadget(asm_source):
    """Compile gadget to raw binary via Docker. Returns (bytes, msg) or (None, error)."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "t.S").write_text(asm_source, encoding="utf-8")

        r = subprocess.run([
            "docker", "run", "--rm", "-v", f"{tmp}:/work", DOCKER_IMAGE,
            "riscv32-unknown-elf-gcc",
            "-march=rv32im_zicsr", "-mabi=ilp32",
            "-ffreestanding", "-nostdlib",
            "-Wl,--no-relax", "-Wl,-Ttext=0x80000000",
            "-o", "/work/t.elf", "/work/t.S",
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr

        r = subprocess.run([
            "docker", "run", "--rm", "-v", f"{tmp}:/work", DOCKER_IMAGE,
            "riscv32-unknown-elf-objcopy", "-O", "binary",
            "/work/t.elf", "/work/t.bin",
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr

        return Path(tmp, "t.bin").read_bytes(), "OK"


# ── Branch decoding ──────────────────────────────────────────────────────────

def decode_branch(word):
    """
    Partially decode a 32-bit RV32 SB-type instruction.
    Returns (branch_name, rs1, rs2, offset) or None if not a branch.
    """
    if (word & 0x7F) != 0x63:   # opcode 0x63 = branch
        return None

    funct3 = (word >> 12) & 0x7
    bname  = {0b000:"beq", 0b001:"bne", 0b100:"blt",
               0b101:"bge", 0b110:"bltu", 0b111:"bgeu"}.get(funct3)
    if bname is None:
        return None

    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F

    # Reconstruct the sign-extended SB immediate.
    imm12  = (word >> 31) & 1
    imm11  = (word >> 7)  & 1
    imm105 = (word >> 25) & 0x3F
    imm41  = (word >> 8)  & 0xF
    imm = (imm12 << 12) | (imm11 << 11) | (imm105 << 5) | (imm41 << 1)
    if imm12:
        imm -= (1 << 13)

    return bname, rs1, rs2, imm


def eval_branch(bname, v1, v2):
    """Evaluate a branch condition. Returns True if the branch is taken."""
    v1u = v1 & 0xFFFFFFFF
    v2u = v2 & 0xFFFFFFFF
    v1s = v1u if v1u < (1 << 31) else v1u - (1 << 32)
    v2s = v2u if v2u < (1 << 31) else v2u - (1 << 32)
    return {
        "beq":  v1u == v2u,
        "bne":  v1u != v2u,
        "blt":  v1s <  v2s,
        "bge":  v1s >= v2s,
        "bltu": v1u <  v2u,
        "bgeu": v1u >= v2u,
    }[bname]


# ── CT-COND trace collection ─────────────────────────────────────────────────

def collect_cond_trace(code, init_vals, window=8):
    """
    Collect a CT-COND observation trace using the always-mispredict model.

    For each branch encountered:
      1. Save architectural state.
      2. Execute the WRONG path for `window` instructions (transient).
      3. Collect transient_pc / transient_mem events.
      4. Restore state and resume on the correct path.
    """
    trace     = []
    transient = [False]

    def hook_code(uc, address, size, user_data):
        trace.append(("transient_pc" if transient[0] else "pc", address))

    def hook_mem(uc, access, address, size, value, user_data):
        trace.append(("transient_mem" if transient[0] else "mem", address))

    try:
        mu = Uc(UC_ARCH_RISCV, UC_MODE_RISCV32)
        mu.mem_map(RAM_BASE, RAM_SIZE)
        mu.mem_write(RAM_BASE, code)
        mu.mem_write(SANDBOX_ADDR, b"\xab\xcd\xef\x01" * (0x1000 // 4))

        for i in TRACKED:
            mu.reg_write(UC_REG[i], init_vals[i])
        mu.reg_write(UC_REG[30], SANDBOX_ADDR)

        mu.hook_add(UC_HOOK_CODE, hook_code)
        mu.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, hook_mem)

        pc  = RAM_BASE
        end = RAM_BASE + len(code)

        while pc < end:
            # Execute exactly one instruction.
            mu.emu_start(pc, end, timeout=0, count=1)
            new_pc = mu.reg_read(UC_RISCV_REG_PC)

            raw     = bytes(mu.mem_read(pc, 4))
            word    = int.from_bytes(raw, "little")
            decoded = decode_branch(word)

            if decoded is not None:
                bname, rs1_idx, rs2_idx, offset = decoded
                v1    = mu.reg_read(UC_REG[rs1_idx])
                v2    = mu.reg_read(UC_REG[rs2_idx])
                taken = eval_branch(bname, v1, v2)

                # Correct path vs wrong path.
                normal_pc = pc + offset if taken else pc + 4
                wrong_pc  = pc + 4      if taken else pc + offset

                # Save state before transient execution.
                saved = {i: mu.reg_read(UC_REG[i]) for i in range(32)}

                # Execute wrong path (speculative window).
                transient[0] = True
                try:
                    if RAM_BASE <= wrong_pc < end:
                        mu.emu_start(wrong_pc, end, timeout=0, count=window)
                except UcError:
                    pass
                transient[0] = False

                # Restore state and continue on correct path.
                for i, v in saved.items():
                    mu.reg_write(UC_REG[i], v)
                mu.reg_write(UC_RISCV_REG_PC, normal_pc)
                pc = normal_pc
            else:
                pc = new_pc if new_pc != pc else pc + 4

    except UcError:
        pass

    return trace


# ── Two-input CT-COND check ──────────────────────────────────────────────────

def make_input_pair(seed):
    """Generate two input states where registers point into the sandbox at different offsets."""
    rng = random.Random(seed)
    state_a = {i: SANDBOX_ADDR + (rng.randrange(0, 0x400) & ~3) for i in TRACKED}
    state_b = {i: SANDBOX_ADDR + (rng.randrange(0, 0x400) & ~3) for i in TRACKED}
    return state_a, state_b


def check_cond(code, seed, window=8):
    """
    Run the CT-COND check for one gadget with two inputs.
    Returns ('safe', ta, tb, []) or ('violation', ta, tb, diffs).
    """
    state_a, state_b = make_input_pair(seed)
    ta = collect_cond_trace(code, state_a, window)
    tb = collect_cond_trace(code, state_b, window)

    min_len = min(len(ta), len(tb))
    diffs   = [(i, ta[i], tb[i]) for i in range(min_len) if ta[i] != tb[i]]

    if diffs or len(ta) != len(tb):
        return "violation", ta, tb, diffs
    return "safe", ta, tb, []


# ── Output ───────────────────────────────────────────────────────────────────

def print_trace(trace, label, max_events=30):
    print(f"\n--- CT-COND trace {label} ---")
    for ev in trace[:max_events]:
        print(f"  [{ev[0]:12s}]  0x{ev[1]:08x}")
    if len(trace) > max_events:
        print(f"  ... ({len(trace)} events total)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="CT-COND checker: Spectre-v1 gadget detection")
    ap.add_argument("--seed",   type=int, default=None)
    ap.add_argument("--runs",   type=int, default=1)
    ap.add_argument("--window", type=int, default=8,
                    help="Speculative window size in instructions (default: 8)")
    args = ap.parse_args()

    base_seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    print(f"CT-COND checker  (window={args.window})")
    print(f"Base seed : {base_seed}   runs: {args.runs}")
    print("=" * 68)

    n_safe = n_viol = n_err = 0
    violations  = []
    last_result = None

    for k in range(args.runs):
        seed = base_seed + k
        random.seed(seed)

        asm, meta = generate_gadget()
        label = f"{meta['instr1']} | {meta['branch']} | {meta['instr2']}"

        code, msg = compile_gadget(asm)
        if code is None:
            print(f"[{seed}] ERROR  {label}  ({msg.strip()[:60]})")
            n_err += 1
            continue

        status, ta, tb, diffs = check_cond(code, seed, args.window)
        last_result = (status, asm, ta, tb, diffs)

        if status == "safe":
            n_safe += 1
            print(f"[{seed}] CT-safe    {label}")
        else:
            n_viol += 1
            violations.append((seed, meta, label, ta, tb, diffs))
            print(f"[{seed}] VIOLATION  {label}")

    # For a single run, show gadget + full traces.
    if args.runs == 1 and last_result is not None:
        status, asm, ta, tb, diffs = last_result
        print(f"\nGadget:\n{asm}")
        print_trace(ta, "A")
        print_trace(tb, "B")

    print("\n" + "=" * 68)
    print(f"runs={args.runs}  CT-safe={n_safe}  violation={n_viol}  error={n_err}")

    for seed, meta, label, ta, tb, diffs in violations:
        print(f"\n{'─' * 68}")
        print(f"VIOLATION  seed={seed}")
        print(f"  Gadget : {label}")
        print(f"  First diverging events ({len(diffs)} total):")
        for i, ea, eb in diffs[:6]:
            print(f"    [{i}] A=({ea[0]:12s}, 0x{ea[1]:08x})  "
                  f"B=({eb[0]:12s}, 0x{eb[1]:08x})")


if __name__ == "__main__":
    main()