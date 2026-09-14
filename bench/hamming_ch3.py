import fpy2 as fp


@fp.fpy(
    meta={
        "name": "NMSE example 3.1",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x >= fp.round(0),
    }
)
def nmse_example_3_1(x: fp.Real) -> fp.Real:
    return fp.sqrt(x + fp.round(1)) - fp.sqrt(x)


@fp.fpy(
    meta={
        "name": "NMSE example 3.3",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
    }
)
def nmse_example_3_3(x: fp.Real, eps: fp.Real) -> fp.Real:
    return fp.sin(x + eps) - fp.sin(x)


@fp.fpy(
    meta={
        "name": "NMSE example 3.4",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x != fp.round(0),
    }
)
def nmse_example_3_4(x: fp.Real) -> fp.Real:
    return (fp.round(1) - fp.cos(x)) / fp.sin(x)


@fp.fpy(
    meta={
        "name": "NMSE example 3.5",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
    }
)
def nmse_example_3_5(N: fp.Real) -> fp.Real:
    return fp.atan(N + fp.round(1)) - fp.atan(N)


@fp.fpy(
    meta={
        "name": "NMSE example 3.6",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x >= fp.round(0),
    }
)
def nmse_example_3_6(x: fp.Real) -> fp.Real:
    return fp.round(1) / fp.sqrt(x) - fp.round(1) / fp.sqrt(x + fp.round(1))


@fp.fpy(
    meta={
        "name": "NMSE problem 3.3.1",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x != fp.round(0),
    },
)
def nmse_problem_3_3_1(x: fp.Real) -> fp.Real:
    return fp.round(1) / (x + fp.round(1)) - fp.round(1) / x


@fp.fpy(
    meta={
        "name": "NMSE problem 3.3.2",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
    }
)
def nmse_problem_3_3_2(x: fp.Real, eps: fp.Real) -> fp.Real:
    return fp.tan(x + eps) - fp.tan(x)


@fp.fpy(
    meta={
        "name": "NMSE problem 3.3.3",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x != fp.round(0) and x != fp.round(1) and x != fp.round(-1),
    }
)
def nmse_problem_3_3_3(x: fp.Real) -> fp.Real:
    return (
        fp.round(1) / (x + fp.round(1))
        - fp.round(2) / x
        + fp.round(1) / (x - fp.round(1))
    )


@fp.fpy(
    meta={
        "name": "NMSE problem 3.3.4",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x >= fp.round(0),
    }
)
def nmse_problem_3_3_4(x: fp.Real) -> fp.Real:
    return fp.pow(x + fp.round(1), fp.round(1) / fp.round(3)) - fp.pow(
        x, fp.round(1) / fp.round(3)
    )


@fp.fpy(
    meta={
        "name": "NMSE problem 3.3.5",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
    }
)
def nmse_problem_3_3_5(x: fp.Real, eps: fp.Real) -> fp.Real:
    return fp.cos(x + eps) - fp.cos(x)


@fp.fpy(
    meta={
        "name": "NMSE problem 3.3.6",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda N: N > fp.round(0),
    }
)
def nmse_problem_3_3_6(N: fp.Real) -> fp.Real:
    return fp.log(N + fp.round(1)) - fp.log(N)


@fp.fpy(
    meta={
        "name": "NMSE problem 3.3.7",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
    }
)
def nmse_problem_3_3_7(x: fp.Real) -> fp.Real:
    return (fp.exp(x) - fp.round(2)) + fp.exp(-x)


@fp.fpy(
    meta={
        "name": "NMSE p42, positive",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda a, b, c: b * b >= fp.round(4) * (a * c) and a != fp.round(0),
    }
)
def nmse_p42_positive(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
    return (-b + fp.sqrt(b * b - fp.round(4) * (a * c))) / (fp.round(2) * a)


@fp.fpy(
    meta={
        "name": "NMSE p42, negative",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda a, b, c: b * b >= fp.round(4) * (a * c) and a != fp.round(0),
    }
)
def nmse_p42_negative(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
    return (-b - fp.sqrt(b * b - fp.round(4) * (a * c))) / (fp.round(2) * a)


@fp.fpy(
    meta={
        "name": "NMSE problem 3.2.1, positive",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda a, b2, c: b2 * b2 >= a * c and a != fp.round(0),
    }
)
def nmse_problem_3_2_1_positive(a: fp.Real, b2: fp.Real, c: fp.Real) -> fp.Real:
    return (-b2 + fp.sqrt(b2 * b2 - a * c)) / a


@fp.fpy(
    meta={
        "name": "NMSE problem 3.2.1, negative",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda a, b2, c: b2 * b2 >= a * c and a != fp.round(0),
    }
)
def nmse_problem_3_2_1_negative(a: fp.Real, b2: fp.Real, c: fp.Real) -> fp.Real:
    return (-b2 - fp.sqrt(b2 * b2 - a * c)) / a


@fp.fpy(
    meta={
        "name": "NMSE example 3.7",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
    }
)
def nmse_example_3_7(x: fp.Real) -> fp.Real:
    return fp.exp(x) - fp.round(1)


@fp.fpy(
    meta={
        "name": "NMSE example 3.8",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda N: N > fp.round(0),
    }
)
def nmse_example_3_8(N: fp.Real) -> fp.Real:
    return ((N + fp.round(1)) * fp.log(N + fp.round(1)) - N * fp.log(N)) - fp.round(1)


@fp.fpy(
    meta={
        "name": "NMSE example 3.9",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x != fp.round(0),
    }
)
def nmse_example_3_9(x: fp.Real) -> fp.Real:
    return fp.round(1) / x - fp.round(1) / fp.tan(x)


@fp.fpy(
    meta={
        "name": "NMSE example 3.10",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "daisy_pre": lambda x: fp.round(-0.99) < x < fp.round(0.99),
        "pre": lambda x: fp.round(-1) < x < fp.round(1),
    }
)
def nmse_example_3_10(x: fp.Real) -> fp.Real:
    return fp.log(fp.round(1) - x) / fp.log(fp.round(1) + x)


@fp.fpy(
    meta={
        "name": "NMSE problem 3.4.1",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x != fp.round(0),
    }
)
def nmse_problem_3_4_1(x: fp.Real) -> fp.Real:
    return (fp.round(1) - fp.cos(x)) / (x * x)


@fp.fpy(
    meta={
        "name": "NMSE problem 3.4.2",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda a, b, eps: eps != fp.round(0),
    }
)
def nmse_problem_3_4_2(a: fp.Real, b: fp.Real, eps: fp.Real) -> fp.Real:
    return (eps * (fp.exp((a + b) * eps) - fp.round(1))) / (
        (fp.exp(a * eps) - fp.round(1)) * (fp.exp(b * eps) - fp.round(1))
    )


@fp.fpy(
    meta={
        "name": "NMSE problem 3.4.3",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda eps: fp.round(-1) < eps < fp.round(1),
    }
)
def nmse_problem_3_4_3(eps: fp.Real) -> fp.Real:
    return fp.log((fp.round(1) - eps) / (fp.round(1) + eps))


@fp.fpy(
    meta={
        "name": "NMSE problem 3.4.4",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x != fp.round(0),
    }
)
def nmse_problem_3_4_4(x: fp.Real) -> fp.Real:
    return fp.sqrt((fp.exp(fp.round(2) * x) - fp.round(1)) / (fp.exp(x) - fp.round(1)))


@fp.fpy(
    meta={
        "name": "NMSE problem 3.4.5",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x != fp.round(0),
    }
)
def nmse_problem_3_4_5(x: fp.Real) -> fp.Real:
    return (x - fp.sin(x)) / (x - fp.tan(x))


@fp.fpy(
    meta={
        "name": "NMSE problem 3.4.6",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x, n: x >= fp.round(0),
    }
)
def nmse_problem_3_4_6(x: fp.Real, n: fp.Real) -> fp.Real:
    return fp.pow(x + fp.round(1), fp.round(1) / n) - fp.pow(x, fp.round(1) / n)


@fp.fpy(
    meta={
        "name": "NMSE section 3.5",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
    }
)
def nmse_section_3_5(a: fp.Real, x: fp.Real) -> fp.Real:
    return fp.exp(a * x) - fp.round(1)


@fp.fpy(
    meta={
        "name": "NMSE section 3.11",
        "cite": ["hamming-1987", "herbie-2015"],
        "fpbench_domain": "textbook",
        "pre": lambda x: x != fp.round(0),
    }
)
def nmse_section_3_11(x: fp.Real) -> fp.Real:
    return fp.exp(x) / (fp.exp(x) - fp.round(1))
