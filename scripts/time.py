import subprocess

from pathlib import Path

import fpy2 as fp

from .options import CompileConfig, EvalConfig

###########################################################
# Compilation

NAME = 'kernel'

HEADER = """\
#include <chrono>
#include <random>
#include <vector>
#include <tuple>
#include <mpfx.hpp>
"""

def monomorphize(ann: fp.ast.TypeAnn, ctx: fp.Context | None):
    """Returns a type annotation monomorphized for the given context."""
    match ann:
        case fp.ast.BoolTypeAnn():
            return fp.types.BoolType()
        case fp.ast.ContextTypeAnn():
            return fp.types.ContextType()
        case fp.ast.RealTypeAnn():
            return fp.types.RealType(ctx)
        case fp.ast.ListTypeAnn():
            return fp.types.ListType(monomorphize(ann.elt, ctx))
        case fp.ast.TupleTypeAnn():
            raise RuntimeError('tuple type annotations are not supported for monomorphization')
        case _:
            raise NotImplementedError(f'cannot monomorphize `{ann}` for context `{ctx}`')

def compile(
    func: fp.Function,
    arg_types: tuple[fp.types.Type, ...],
    compile_config: CompileConfig
) -> str:
    """Compiles an FPy function for given contexts."""
    compiler = fp.MPFXCompiler()
    code = compiler.compile(func, name=NAME, arg_types=arg_types, elim_round=compile_config.elim_round)
    return code

###########################################################
# Test harness

def make_quantizer(compiler: fp.MPFXCompiler, ty: fp.types.RealType) -> str:
    if ty.ctx is None or isinstance(ty.ctx, fp.utils.NamedId):
        raise RuntimeError('Cannot generate inputs for real type with no context')
    return compiler.compile_context(ty.ctx)

def emit_distribution(lines: list[str], compiler: fp.MPFXCompiler, ty: fp.types.RealType, idx: int):
    if ty.ctx is not None and isinstance(ty.ctx, fp.ExpContext):
        min_exp = ty.ctx.minval().e
        max_exp = ty.ctx.maxval().e
        lines.append(f'    std::uniform_int_distribution<int> dist{idx}({min_exp}, {max_exp});')
        return f'std::ldexp(1.0, dist{idx}(rng))'
    else:
        ctx_expr = make_quantizer(compiler, ty)
        lines.append(f'    const auto ctx{idx} = {ctx_expr};')
        lines.append(f'    std::uniform_real_distribution<double> dist{idx}(-1.0, 1.0);')
        return f'mpfx::round<mpfx::Flags::NO_FLAGS>(dist{idx}(rng), ctx{idx})'

def test_harness(func: fp.Function, arg_types: tuple[fp.types.Type, ...]) -> str:
    """Emits a test harness for FPy functions."""
    compiler = fp.MPFXCompiler()

    lines = [
        'template <typename T>',
        'inline void DoNotOptimizeAway(const T& value) {',
        '    asm volatile("" : : "r,m"(value) : "memory");',
        '}',
        '',
        'int main(int argc, char** argv) {'
    ]

    # benchmark takes three arguments from the command line:
    # - number of inputs
    # - size of vectors
    # - seeds for randomness
    lines.append('    if (argc < 4) {')
    lines.append('        std::cerr << "Usage: " << argv[0] << " <num_inputs> <vector_size> <seed>" << std::endl;')
    lines.append('        return 1;')
    lines.append('    }')
    lines.append('    const size_t num_inputs = std::stoul(argv[1]);')
    lines.append('    const size_t vector_size = std::stoul(argv[2]);')
    lines.append('    const size_t seed = std::stoul(argv[3]);')
    lines.append('')

    # create a random device
    lines.append('    std::mt19937 rng(seed);')
    lines.append('')

    # for each argument, seed a vector of inputs
    for i, ty in enumerate(arg_types):
        ty_str = compiler.compile_type(ty).to_cpp()
        lines.append(f'    std::vector<{ty_str}> arg{i}(num_inputs);')

        match ty:
            case fp.types.RealType():
                # create a random uniform distribution
                sampler = emit_distribution(lines, compiler, ty, i)
                lines.append(f'    for (size_t i = 0; i < num_inputs; ++i) {{')
                lines.append(f'        arg{i}[i] = {sampler};')
                lines.append('    }')
                lines.append('')
            case fp.types.ListType() if isinstance(ty.elt, fp.types.RealType):
                # create a random uniform distribution
                sampler = emit_distribution(lines, compiler, ty.elt, i)
                lines.append(f'    for (size_t i = 0; i < num_inputs; ++i) {{')
                lines.append(f'        arg{i}[i] = {ty_str}(vector_size);')
                lines.append(f'        for (size_t j = 0; j < vector_size; ++j) {{')
                lines.append(f'            arg{i}[i][j] = {sampler};')
                lines.append('        }')
                lines.append('    }')
                lines.append('')
            case _:
                raise NotImplementedError(f'input generation not implemented for type `{ty}`')

    ty_str = compiler.compile_type(ty).to_cpp()
    arg_list = ', '.join(f'arg{j}[i]' for j in range(len(arg_types)))

    # execute kernel in a loop with timing
    lines.append('    auto start = std::chrono::steady_clock::now();')
    lines.append('    for (size_t i = 0; i < num_inputs; ++i) {')
    lines.append(f'        auto result = {NAME}({arg_list});')
    lines.append('        DoNotOptimizeAway(result);')
    lines.append('    }')
    lines.append('    auto end = std::chrono::steady_clock::now();')

    # measure time in nanoseconds
    lines.append('    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();')
    lines.append('    std::cout << duration << "\\n";')
    lines.append('')

    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines)


###########################################################
# Compile the benchmark

CPP_OPTS = ['-O3', '-march=native', '-mtune=native', '-std=c++20']

def compile_benchmark(path: Path) -> Path:
    """Compiles the benchmark at the given path."""
    # replace .cpp with executable name
    binary_path = path.with_suffix('')

    # compile command
    cmd = ['c++', *CPP_OPTS, str(path), '-o', str(binary_path), '-lmpfx']
    print(f'  compiling benchmark [cmd: {" ".join(cmd)}]')
    subprocess.run(cmd, check=True)

    return binary_path


###########################################################
# Code

def time_benchmark(
    func: fp.Function,
    ctxs: tuple[fp.Context | None, ...],
    key: int,
    config: EvalConfig,
    compile_config: CompileConfig
) -> float:
    """Times the execution of an FPy function over a number of iterations."""
    print(f' Benchmarking function `{func.name}` with config={compile_config}...')

    # apply monomorphization
    arg_types = tuple(monomorphize(arg.type, ctx) for arg, ctx in zip(func.args, ctxs))

    # compile the function
    code = compile(func, arg_types, compile_config)

    # generate the harness
    harness = test_harness(func, arg_types)

    # output directory for this benchmark
    output_dir = config.output_dir / f'benchmark_{key}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'  working files under `{output_dir}`')

    # write code into the output directory
    output_file = output_dir / 'harness.cpp'
    with open(output_file, 'w') as f:
        # emit metadata for bookkeeping
        print('// Auto-generated benchmark code ', file=f)
        print(f'// function: {func.name}', file=f)
        print(f'// contexts:', file=f)
        for ctx in ctxs:
            print(f'//   {ctx}', file=f)
        print('// definition:', file=f)
        for line in func.format().splitlines():
            print(f'//   {line}', file=f)
        print('', file=f)

        # includes
        f.write(HEADER)
        print('', file=f)

        # write compiled code
        f.write(code)
        f.write('\n\n')

        # emit test harness
        f.write(harness)

    print(f'  benchmark written to `{output_file}`.')

    # compile the benchmark
    binary_path = compile_benchmark(output_file)
    print(f'  compiled binary to `{binary_path}`.')

    # run harness with parameters and capture output
    cmd = [str(binary_path), str(config.num_inputs), '32', str(config.seed)]
    print(f'  executing benchmark [cmd: {" ".join(cmd)}]...')
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # parse elapsed time from output
    for line in result.stdout.splitlines():
        try:
            duration_ns = int(line.strip())
            duration_s = duration_ns / 1_000_000_000.0
            print(f'  benchmark completed in {duration_s:.6f}s.')
            return duration_s
        except ValueError:
            continue

    return 0.0
