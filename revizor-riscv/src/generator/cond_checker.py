"""
CT-COND contract checker: Constant-Time contract with Conditional branch speculation.

Based on Eric Garcia's TFG section 3.3 and Guarnieri et al. 2021.

The CT-COND contract states:
  Two executions of the same program with different inputs must produce the
  same observation trace even when considering speculative execution on
  conditional branch mispredictions.

  Trace CT-COND = trace CT-SEQ + observations during transient execution
                  (the wrong branch path, executed speculatively).

Spectre variant 1 (PHT) pattern detected:
  beq  x8, x0, .skip     ← branch on secret-dependent value
  lw   x9, 0(x8)         ← transient load, address depends on secret → LEAK
.skip:
  nop

Usage:
  python3 cond_checker.py                    # one random test
  python3 cond_checker.py --runs 50          # fuzz 50 cases
  python3 cond_checker.py --seed 42          # reproducible
  python3 cond_checker.py --window 5         # speculative window size
"""
import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path
from unicorn import Uc, UC_ARCH_RISCV, UC_MODE_RISCV32, UcError
from unicorn import UC_HOOK_CODE, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE
from unicorn.riscv_const import *

RANGE_RE = re.compile(r"^\[(-?\d+)-(-?\d+)\]$")

DOCKER_IMAGE  = "proteus"
RAM_BASE      = 0x80000000
SANDBOX_ADDR  = 0x80001000
RAM_SIZE      = 0x00100000      # 1 MiB

# Registers used in generated gadgets
RESERVED_REGS = {"x30", "x31"}
TRACKED       = list(range(5, 30))
UC_REG = {i: globals()[f"UC_RISCV_REG_X{i}"] for i in range(32)}

# Branch instructions and their COND (condition to evaluate)
BRANCHES = ["BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"]


# ---------------------------------------------------------------------------
# Instruction generation helpers (reused from contract_checker.py)
# ---------------------------------------------------------------------------
def pick_register(operand_values, exclude=None):
    candidates = [v for v in operand_values if v not in (exclude or set())]
    if not candidates:
        raise ValueError(f"No registers left after excluding {exclude}")
    return random.choice(candidates)


def pick_immediate(lo, hi, align=1):
    v = random.randint(lo // align, hi // align) * align
    return v


def rand_reg(exclude=None):
    """Pick a random general-purpose register (x5..x29), excluding reserved."""
    exclude = (exclude or set()) | RESERVED_REGS
    return random.choice([f"x{i}" for i in TRACKED if f"x{i}" not in exclude])


# ---------------------------------------------------------------------------
# Gadget generation: a 3-instruction Spectre-v1 pattern
#
# Structure:
#   instr_1  : arithmetic/logic that produces a secret-dependent value in rd1
#   branch   : conditional branch on rd1
#   instr_2  : memory access (load) whose address depends on rd1 (transient)
#   .skip    : landing pad (nop)
# ---------------------------------------------------------------------------

def generate_gadget():
    rd1  = rand_reg()
    rs_a = rand_reg(exclude={rd1})
    rs_b = rand_reg(exclude={rd1, rs_a})

    arith_ops = ["add", "sub", "and", "or", "xor", "mul"]
    op1    = random.choice(arith_ops)
    instr1 = f"{op1} {rd1}, {rs_a}, {rs_b}"

    # Pour les comparaisons d'ordre, utiliser rs_b comme second opérande
    # (valeur aléatoire connue), pas x0 qui est toujours 0
    branch_ops = {
        "beq":  f"beq  {rd1}, x0, .skip",
        "bne":  f"bne  {rd1}, x0, .skip",
        "blt":  f"blt  {rd1}, {rs_b}, .skip",
        "bge":  f"bge  {rd1}, {rs_b}, .skip",
        "bltu": f"bltu {rd1}, {rs_b}, .skip",
        "bgeu": f"bgeu {rd1}, {rs_b}, .skip",
    }
    branch_name  = random.choice(list(branch_ops.keys()))
    branch_instr = branch_ops[branch_name]

    rd2    = rand_reg(exclude={rd1, rs_a, rs_b})
    load_ops = ["lw", "lh", "lb", "lbu", "lhu"]
    op2    = random.choice(load_ops)
    instr2 = f"{op2} {rd2}, 0({rd1})"

    asm = f""".section .text
.globl _start
_start:
    lui   x30, 0x80001

    {instr1}

    {branch_instr}

    # transient load: address depends on rd1 (secret-dependent)
    {instr2}

.skip:
    nop
"""
    meta = {
        "instr1":  instr1,
        "branch":  branch_instr,
        "instr2":  instr2,
        "rd1":     rd1,
        "rs_a":    rs_a,
        "rs_b":    rs_b,
    }
    return asm, meta

# ---------------------------------------------------------------------------
# Compile via Docker
# ---------------------------------------------------------------------------
def compile_gadget(asm_source):
    """Compile to raw binary via Docker. Returns bytes or None."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "t.S").write_text(asm_source)
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


# ---------------------------------------------------------------------------
# CT-COND trace collection
#
# Strategy: "always mispredict" (conservative, same as Revizor default).
# For each conditional branch encountered:
#   1. Record the actual branch outcome (taken / not-taken).
#   2. Force Unicorn to take the OPPOSITE path for `window` instructions.
#   3. Collect observations during that transient window.
#   4. Restore architectural state and resume normal execution.
# ---------------------------------------------------------------------------
BRANCH_OPCODES = {
    # opcode = bits[6:0] = 0b1100011 = 0x63
    # funct3 distinguishes BEQ/BNE/BLT/BGE/BLTU/BGEU
    0x63: {
        0b000: "beq",
        0b001: "bne",
        0b100: "blt",
        0b101: "bge",
        0b110: "bltu",
        0b111: "bgeu",
    }
}


def decode_branch(word):
    """
    Partially decode a 32-bit RV32 SB-type instruction.
    Returns (branch_name, rs1, rs2, offset) or None if not a branch.
    """
    opcode = word & 0x7F
    if opcode != 0x63:
        return None
    funct3 = (word >> 12) & 0x7
    bname  = BRANCH_OPCODES[0x63].get(funct3)
    if bname is None:
        return None
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    # decode SB immediate (sign-extended)
    imm12  = (word >> 31) & 1
    imm11  = (word >> 7)  & 1
    imm105 = (word >> 25) & 0x3F
    imm41  = (word >> 8)  & 0xF
    imm = (imm12 << 12) | (imm11 << 11) | (imm105 << 5) | (imm41 << 1)
    if imm12:
        imm -= (1 << 13)
    return bname, rs1, rs2, imm


def eval_branch(bname, v1, v2):
    """Evaluate branch condition. Returns True if branch is taken."""
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

def collect_cond_trace(code, init_vals, window=8):
    trace     = []
    transient = [False]

    def hook_code(uc, address, size, user_data):
        kind = "transient_pc" if transient[0] else "pc"
        trace.append((kind, address))

    def hook_mem(uc, access, address, size, value, user_data):
        kind = "transient_mem" if transient[0] else "mem"
        trace.append((kind, address))

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
            # execute exactly one instruction
            mu.emu_start(pc, end, timeout=0, count=1)

            # read actual PC after execution
            new_pc = mu.reg_read(UC_RISCV_REG_PC)

            # read the instruction bytes at pc to detect branches
            raw  = bytes(mu.mem_read(pc, 4))
            word = int.from_bytes(raw, "little")
            decoded = decode_branch(word)

            if decoded is not None:
                bname, rs1_idx, rs2_idx, offset = decoded
                v1    = mu.reg_read(UC_REG[rs1_idx])
                v2    = mu.reg_read(UC_REG[rs2_idx])
                taken = eval_branch(bname, v1, v2)

                normal_pc = pc + offset if taken else pc + 4
                wrong_pc  = pc + 4      if taken else pc + offset

                # save state before transient execution
                saved = {i: mu.reg_read(UC_REG[i]) for i in range(32)}
                saved_pc = normal_pc

                # simulate wrong path (speculative window)
                transient[0] = True
                try:
                    if RAM_BASE <= wrong_pc < end:
                        mu.emu_start(wrong_pc, end, timeout=0, count=window)
                except UcError:
                    pass
                transient[0] = False

                # restore state and continue on correct path
                for i, v in saved.items():
                    mu.reg_write(UC_REG[i], v)
                mu.reg_write(UC_RISCV_REG_PC, normal_pc)
                pc = normal_pc
            else:
                # not a branch: advance to next instruction
                pc = new_pc if new_pc != pc else pc + 4

    except UcError:
        pass

    return trace

# ---------------------------------------------------------------------------
# Two-input CT-COND check
# ---------------------------------------------------------------------------
def make_input_pair(base_seed):
    """Two states where all registers point into the sandbox at different offsets."""
    rng = random.Random(base_seed)
    state_a, state_b = {}, {}
    for i in TRACKED:
        state_a[i] = SANDBOX_ADDR + (rng.randrange(0, 0x400) & ~3)
        state_b[i] = SANDBOX_ADDR + (rng.randrange(0, 0x400) & ~3)
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
    diffs = [(i, ta[i], tb[i]) for i in range(min_len) if ta[i] != tb[i]]

    if diffs or len(ta) != len(tb):
        return "violation", ta, tb, diffs
    return "safe", ta, tb, []


# ---------------------------------------------------------------------------
# Pretty print a trace
# ---------------------------------------------------------------------------
def print_trace(trace, label, max_events=30):
    print(f"\n--- CT-COND trace {label} ---")
    for ev in trace[:max_events]:
        tag = f"[{ev[0]:12s}]"
        print(f"  {tag}  0x{ev[1]:08x}")
    if len(trace) > max_events:
        print(f"  ... ({len(trace)} events total)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CT-COND checker: Spectre-v1 detection")
    ap.add_argument("--seed",   type=int, default=None)
    ap.add_argument("--runs",   type=int, default=1)
    ap.add_argument("--window", type=int, default=8,
                    help="Speculative window size (instructions after branch)")
    args = ap.parse_args()

    base_seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    print(f"CT-COND checker  (window={args.window})")
    print(f"Base seed : {base_seed}   runs: {args.runs}")
    print("=" * 68)

    n_safe = n_viol = n_err = 0
    violations = []

    for k in range(args.runs):
        seed = base_seed + k
        random.seed(seed)

        asm, meta = generate_gadget()
        label = f"{meta['instr1']} | {meta['branch'].strip()} | {meta['instr2']}"

        try:
            code, msg = compile_gadget(asm)
        except Exception as e:
            print(f"[{seed}] ERROR (compile exception: {e})")
            n_err += 1
            continue

        if code is None:
            print(f"[{seed}] ERROR (compile failed)")
            n_err += 1
            continue

        status, ta, tb, diffs = check_cond(code, seed, args.window)

        if status == "safe":
            n_safe += 1
            print(f"[{seed}] CT-safe    {label}")
        else:
            n_viol += 1
            violations.append((seed, meta, label, ta, tb, diffs))
            print(f"[{seed}] VIOLATION  {label}")

    # detail for single run
    if args.runs == 1:
        random.seed(base_seed)
        asm, meta = generate_gadget()
        code, _ = compile_gadget(asm)
        if code:
            state_a, state_b = make_input_pair(base_seed)
            ta = collect_cond_trace(code, state_a, args.window)
            tb = collect_cond_trace(code, state_b, args.window)
            print(f"\nGadget:\n{asm}")
            print_trace(ta, "A")
            print_trace(tb, "B")

    print("\n" + "=" * 68)
    print(f"runs={args.runs}  CT-safe={n_safe}  violation={n_viol}  error={n_err}")

    for seed, meta, label, ta, tb, diffs in violations:
        print(f"\n{'─'*68}")
        print(f"VIOLATION  seed={seed}")
        print(f"  Gadget : {label}")
        print(f"  First diverging events ({len(diffs)} total):")
        for i, ea, eb in diffs[:6]:
            print(f"    [{i}] A=({ea[0]:12s}, 0x{ea[1]:08x})  "
                  f"B=({eb[0]:12s}, 0x{eb[1]:08x})")


if __name__ == "__main__":
    main()